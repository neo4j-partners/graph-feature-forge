# Feature Engineering

The enrichment pipeline can only analyze customers who have profile documents, and in this dataset only 3 of 103 customers do. The other 100 customers exist in the graph with holdings, accounts, and connections, but no risk profile. The feature engineering stage closes that gap by extracting structural signal from the full graph topology, training classifiers on the labeled subset, and predicting risk profiles for everyone else.

Three entry points run sequentially as Databricks jobs. Each builds on the previous one, and together they form a three-way experiment that measures how much value graph structure adds over tabular features alone.

## Entry Points

### gds_fastrp_features.py

Establishes the baseline graph model. Projects the portfolio graph in Neo4j GDS, computes FastRP embeddings that encode each customer's structural neighborhood into a dense vector, exports those vectors to a Delta feature table, trains an AutoML classifier, and writes predictions back to Neo4j.

* **Graph projection**: Projects Customer, Account, Position, Stock, and Company nodes with four undirected relationship types (HAS_ACCOUNT, HAS_POSITION, OF_SECURITY, OF_COMPANY) plus any enrichment relationships discovered from the enrichment log Delta table
* **FastRP computation**: Runs Fast Random Projection with 128 dimensions, iteration weights [0.0, 1.0, 1.0, 0.8, 0.5] for multi-hop diffusion, and a fixed seed of 42 for determinism; writes `fastrp_embedding` to all nodes in Neo4j
* **Louvain computation**: Runs Louvain community detection with 10 max levels and 10 max iterations; writes `community_id` to all nodes (computed here but excluded from this model's features)
* **Feature export**: Reads Customer nodes via the Neo4j Spark Connector, explodes the 128-dimensional embedding array into individual columns (`fastrp_0` through `fastrp_127`), appends `annual_income` and `credit_score`, and writes the result to `customer_graph_features` in Delta
* **Holdout creation**: Performs stratified sampling that keeps 10 labeled examples per risk category and nulls the rest, simulating a sparse-label scenario; saves the original labels to a `holdout_ground_truth` table that all three experiments share
* **AutoML training**: Runs Databricks AutoML classification optimizing F1 for up to 30 minutes under the `/Shared/graph-feature-forge/fastrp_risk_classification` experiment; registers the best model in Unity Catalog with the `@Champion` alias
* **Scoring and writeback**: Loads the Champion model as a Spark UDF, scores every customer with a null `risk_category`, evaluates accuracy against the held-out ground truth, and writes `risk_category_predicted`, `prediction_source`, and `prediction_timestamp` back to the Customer nodes in Neo4j

### gds_community_features.py

Tests whether Louvain community detection improves over embeddings alone. Adds `community_id` as a categorical feature alongside the FastRP dimensions, retrains AutoML on the combined feature set, and conditionally promotes the new model to Champion only if F1 improves.

* **Feature computation**: Runs the same graph projection, FastRP, and Louvain pipeline as the first entry point but retains the GDS graph object in memory for kNN analysis
* **Feature export with community**: Exports the same feature table but includes `community_id` as an additional string categorical column
* **Holdout reapplication**: Reads the `holdout_ground_truth` table created by the first entry point and re-nulls the exact same customer records, ensuring the validation set is identical across experiments
* **AutoML training with conditional promotion**: Trains under the `/Shared/graph-feature-forge/fastrp_louvain_risk_classification` experiment; compares the new best F1 against the current Champion's F1 and promotes the new version only if it exceeds it
* **kNN nearest-neighbor analysis**: Runs kNN on FastRP embeddings with top-K of 5 to create `SIMILAR_TO` relationships; analyzes spotlight customers (James Anderson, Maria Rodriguez, Robert Chen) to show how community membership correlates with embedding similarity

### gds_baseline_comparison.py

Trains a tabular-only model that excludes all graph features and produces a three-way comparison across all experiments. This entry point validates that graph structure contributes predictive value beyond what `annual_income` and `credit_score` provide on their own.

* **Tabular baseline training**: Reads the existing feature table and dynamically identifies all graph-derived columns (`fastrp_0` through `fastrp_127` and `community_id`) to exclude; trains AutoML under `/Shared/graph-feature-forge/tabular_only_baseline` using only the two tabular features
* **Three-way experiment comparison**: Queries MLflow for the best F1 and model type from each of the three experiments and produces a formatted comparison table
* **Feature importance analysis**: Extracts feature importances from the best FastRP + Louvain model, reports the top 20 features, and counts how many are graph-derived versus tabular

## Architecture

### From Graph Topology to ML Features

The core insight is that customers who hold similar portfolios through similar account structures occupy similar neighborhoods in the graph. Two customers who both hold tech stocks through brokerage accounts at the same bank are structurally closer than two customers who happen to share the same income bracket. FastRP captures this proximity by propagating random signals through the graph's edges and collecting the aggregated signal at each node into a fixed-length vector.

The graph projection is the first step. GDS creates an in-memory copy of the subgraph defined by five node labels and four relationship types. All relationships are projected as undirected because the direction of ownership (customer owns account versus account belongs to customer) is less important than the connection itself. When the enrichment pipeline has already run, the projection dynamically includes any new relationship types discovered during enrichment, so the feature space grows as the graph compounds knowledge.

FastRP then runs on the projected graph with five iteration weights: [0.0, 1.0, 1.0, 0.8, 0.5]. The zero at position zero means a node's own initial random vector is discarded; all signal comes from neighbors. Weights at positions one and two carry full strength from immediate and two-hop neighbors. The decay at positions three and four means that three-hop and four-hop influence contributes less, preventing distant noise from diluting the local structure. The result is a 128-dimensional vector on every node that encodes multi-hop structural context.

Louvain community detection runs on the same projection. Where FastRP produces a continuous embedding, Louvain produces a discrete community assignment by iteratively optimizing modularity. Customers in the same community share denser internal connections than connections to the rest of the graph. This provides a complementary signal: FastRP captures fine-grained structural similarity while Louvain captures coarse-grained cluster membership.

### Feature Export and the Sparse Label Problem

The Neo4j Spark Connector reads Customer nodes with their computed properties into a Spark DataFrame. The embedding array (which may arrive as a native array or a JSON string depending on the connector version) is parsed and exploded into 128 individual double columns. Together with `annual_income`, `credit_score`, and optionally `community_id`, this forms the feature table written to Delta.

Only 3 of 103 customers have a `risk_profile` label. To evaluate model quality without external labels, the pipeline simulates a holdout: it keeps 10 labeled examples per risk category and nulls the rest, then trains on the sparse set and evaluates against the ground truth it set aside. The ground truth table records which customers were held out with a boolean flag, and subsequent experiments reapply the identical split by joining on that flag. This guarantees that all three models (FastRP-only, FastRP + Louvain, tabular-only) are evaluated against the same validation set.

### AutoML Training and Model Lifecycle

Each experiment feeds its feature table to Databricks AutoML, which searches across classifier families (logistic regression, random forest, gradient boosted trees, and others) optimizing F1. The 30-minute timeout gives AutoML enough room to try dozens of configurations without consuming excessive cluster time.

The first entry point registers its best model as Champion immediately since no prior version exists. The second entry point takes a more cautious approach: it registers a new model version, fetches the current Champion's F1 from its MLflow run metrics, and promotes the new version only if it strictly improves. This prevents the community-augmented model from regressing the production model if Louvain adds noise rather than signal.

All models register under a single Unity Catalog name (`graph_feature_forge_risk_classifier`) with the `@Champion` alias tracking the production version. The scoring step loads whichever version currently holds that alias, so the writeback always uses the best available model regardless of which experiment produced it.

### Scoring and Neo4j Writeback

The Champion model is loaded as a Spark UDF via `mlflow.pyfunc.spark_udf()`, which distributes inference across the cluster. The pipeline filters the feature table to customers with null `risk_category`, applies the UDF across all feature columns, and collects predictions. Each prediction is written back to Neo4j as a `risk_category_predicted` property on the Customer node, alongside a `prediction_source` identifier and a UTC timestamp. The Spark Connector handles the writeback via merge on `customer_id`, so repeated runs update rather than duplicate.

This closes the loop: the enrichment pipeline now has risk profiles for all 103 customers, not just the 3 with documents. The next enrichment cycle can use those predictions as additional context when synthesizing gap analyses, and future GDS projections reflect an increasingly annotated graph.

### Three-Way Comparison

The baseline entry point is the control experiment. By dynamically excluding every graph-derived column and training on only `annual_income` and `credit_score`, it establishes the ceiling for what tabular features can achieve. The three-way comparison queries each experiment's MLflow runs for the best F1 score and model type, producing a table that directly answers whether graph features improve classification.

Feature importance extraction from the best graph-augmented model reveals which of the 128 FastRP dimensions carry the most predictive weight and whether `community_id` ranks among the top features. If a majority of the top-20 features are graph-derived, the architecture has demonstrated that structural signal from the portfolio graph encodes information that income and credit score alone cannot capture.

### Net Outcome

The pipeline transforms a sparsely labeled knowledge graph into a fully scored customer base. Three customers with documents become 103 customers with risk predictions. The predictions flow back into Neo4j where they compound with the enrichment pipeline's discoveries: customers with predicted risk profiles give the gap analysis richer context, and the GDS projection in the next cycle operates on a denser, more annotated graph. Each run of the feature engineering stage both consumes the graph's current state and improves it for the next iteration.
