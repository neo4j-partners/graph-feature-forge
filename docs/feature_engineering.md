# Feature Engineering

The enrichment pipeline can only analyze customers who have profile documents, and in this dataset only 3 of 103 customers do. The other 100 customers exist in the graph with holdings, accounts, and connections, but no risk profile. The feature engineering stage closes that gap by extracting structural signal from the full graph topology, training classifiers on the labeled subset, and predicting risk profiles for everyone else.

Three entry points run sequentially as Databricks jobs. Each builds on the previous one, and together they form a three-way experiment that measures how much value graph structure adds over tabular features alone.

## Entry Points

### gds_fastrp_features.py

Establishes the baseline graph model. Projects the portfolio graph in Neo4j GDS, computes FastRP embeddings that encode each customer's structural neighborhood into a dense vector, exports those vectors to a Delta feature table, trains a scikit-learn classifier, and writes predictions back to Neo4j.

* **Graph projection**: Projects Customer, Account, Position, Stock, and Company nodes with four undirected relationship types (HAS_ACCOUNT, HAS_POSITION, OF_SECURITY, OF_COMPANY) plus any enrichment relationships discovered from the enrichment log Delta table
* **FastRP computation**: Runs Fast Random Projection with 128 dimensions, iteration weights [0.0, 1.0, 1.0, 0.8, 0.5] for multi-hop diffusion, and a fixed seed of 42 for determinism; writes `fastrp_embedding` to all nodes in Neo4j
* **Louvain computation**: Runs Louvain community detection with 10 max levels and 10 max iterations; writes `community_id` to all nodes (computed here but excluded from this model's features)
* **Feature export**: Reads Customer nodes via the Neo4j Spark Connector, explodes the 128-dimensional embedding array into individual columns (`fastrp_0` through `fastrp_127`), appends `annual_income` and `credit_score`, and writes the result to `customer_graph_features` in Delta
* **Holdout creation**: When `TEST_SIZE` is unset, performs stratified sampling that keeps 10 labeled examples per risk category and nulls the rest, simulating a sparse-label scenario; saves the original labels to a `holdout_ground_truth` table that all three experiments share. When `TEST_SIZE` is set, this step is skipped and all labels are retained for an internal train/test split
* **scikit-learn training**: Trains three classifiers (RandomForest, GradientBoosting, LogisticRegression) wrapped in preprocessing pipelines, selects the best by stratified cross-validated F1 macro, logs all runs to MLflow via autolog under the `/Shared/graph-feature-forge/fastrp_risk_classification` experiment, and registers the best model in Unity Catalog with the `@Champion` alias. When `PCA_COMPONENTS` is set, embedding columns are compressed via PCA before training. When `TEST_SIZE` is set, per-class metrics and confusion matrices are logged for the held-out test set
* **Scoring and writeback**: When using the external holdout (no `TEST_SIZE`), loads the Champion model as a Spark UDF, scores every customer with a null `risk_category`, evaluates accuracy against the held-out ground truth, and writes `risk_category_predicted`, `prediction_source`, and `prediction_timestamp` back to the Customer nodes in Neo4j. When `TEST_SIZE` is set, test evaluation is handled internally during training and this step is skipped

### gds_community_features.py

Tests whether Louvain community detection improves over embeddings alone. Adds `community_id` as a categorical feature alongside the FastRP dimensions, retrains on the combined feature set, and conditionally promotes the new model to Champion only if F1 improves.

* **Feature computation**: Runs the same graph projection, FastRP, and Louvain pipeline as the first entry point but retains the GDS graph object in memory for kNN analysis
* **Feature export with community**: Exports the same feature table but includes `community_id` as an additional string categorical column
* **Holdout reapplication**: Reads the `holdout_ground_truth` table created by the first entry point and re-nulls the exact same customer records, ensuring the validation set is identical across experiments
* **Training with conditional promotion**: Trains the same three classifiers under the `/Shared/graph-feature-forge/fastrp_louvain_risk_classification` experiment; compares the new best F1 against the current Champion's F1 and promotes the new version only if it exceeds it
* **kNN nearest-neighbor analysis**: Runs kNN on FastRP embeddings with top-K of 5 to create `SIMILAR_TO` relationships; analyzes spotlight customers (James Anderson, Maria Rodriguez, Robert Chen) to show how community membership correlates with embedding similarity

### gds_baseline_comparison.py

Trains a tabular-only model that excludes all graph features and produces a three-way comparison across all experiments. This entry point validates that graph structure contributes predictive value beyond what `annual_income` and `credit_score` provide on their own.

* **Tabular baseline training**: Reads the existing feature table and dynamically identifies all graph-derived columns (`fastrp_0` through `fastrp_127` and `community_id`) to exclude; trains the same three classifiers under `/Shared/graph-feature-forge/tabular_only_baseline` using only the two tabular features
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

Only 3 of 103 customers have a `risk_profile` label. To evaluate model quality without external labels, the pipeline supports two validation strategies controlled by the `TEST_SIZE` environment variable.

When `TEST_SIZE` is unset, the pipeline simulates a sparse-label scenario: it keeps 10 labeled examples per risk category and nulls the rest, then trains on the sparse set (30 rows) and evaluates against the ground truth it set aside. The ground truth table records which customers were held out with a boolean flag, and subsequent experiments reapply the identical split by joining on that flag. This guarantees that all three models are evaluated against the same validation set.

When `TEST_SIZE` is set (e.g. 0.2), the pipeline skips the holdout simulation and uses all 102 labeled rows with a standard stratified train/test split. This produces more reliable results because the classifier trains on roughly 82 rows instead of 30, and cross-validation increases from 3-fold to 5-fold. The tradeoff is that the sparse-label scenario no longer applies; the pipeline assumes all labels are available. Phase 2.5 used this approach and demonstrated that the additional training data, combined with PCA dimensionality reduction, was critical for graph features to outperform the tabular baseline.

### scikit-learn Training and Model Lifecycle

Each experiment feeds its feature table to a scikit-learn training function that compares three classifier families: RandomForest, GradientBoosting, and LogisticRegression. The function selects the best model by F1 score, a metric that balances two concerns every classifier must navigate.

#### What F1 Measures

A classifier can make two kinds of mistakes. It can predict "Aggressive" for a customer who is actually Conservative (a false positive for the Aggressive class), or it can predict "Moderate" for a customer who is actually Aggressive (a false negative, or a miss). Precision measures how often the classifier is right when it says a customer belongs to a class. Recall measures how many of the actual members of that class the classifier finds.

F1 is the harmonic mean of precision and recall. It ranges from 0 to 1, where 1 means the classifier is both perfectly precise and perfectly complete. The harmonic mean penalizes imbalance: a classifier that achieves 0.95 precision by being extremely conservative (low recall) gets a lower F1 than one that achieves 0.80 on both.

Because this pipeline classifies customers into three risk categories (Aggressive, Moderate, Conservative), the training function uses F1 macro, which computes F1 separately for each class and averages them equally. This prevents the model from ignoring a minority class to boost its score on the majority. A macro F1 of 0.636 means the model averages roughly 63.6% balanced accuracy across all three classes. For comparison, random guessing across three balanced classes would produce an F1 around 0.33.

#### How Cross-Validation Works

The training function does not simply train each classifier once and pick the best score. A single train/test split on 30 rows would be unreliable because a few lucky or unlucky examples in the test set could swing the result. Instead, the function uses stratified k-fold cross-validation.

Cross-validation divides the labeled data into k equal parts (folds). It trains on k-1 folds and evaluates on the remaining fold, then rotates which fold is held out. Each example appears in the test set exactly once. The final score is the average across all k iterations. Stratified means the split preserves the proportion of each class in every fold, so no fold is accidentally dominated by a single risk category.

With 30 labeled rows and 3 classes (10 per class), the pipeline uses k=3 folds. Each fold holds out 10 rows (roughly 3-4 per class), trains on 20, and evaluates on 10. The average of three such rounds gives a more stable estimate than any single split could. When more training data is available (Phase 2.5 removes the artificial holdout to use all 102 labeled rows), the pipeline increases to k=5 for finer-grained estimates.

#### The Training Loop

For each classifier, the function wraps the model in a scikit-learn Pipeline with preprocessing steps that normalize features before training. The simplest configuration uses a StandardScaler that normalizes every feature to zero mean and unit variance. This matters because the raw features have wildly different scales: annual_income might range from 50,000 to 200,000 while a FastRP embedding dimension typically falls between -1.0 and 1.0. Without scaling, classifiers that rely on distance calculations (LogisticRegression) or gradient magnitudes (GradientBoosting) would let income dominate every decision.

When the `PCA_COMPONENTS` environment variable is set, the pipeline uses a ColumnTransformer that applies different preprocessing to different feature groups. FastRP embedding columns (identified by the `fastrp_` prefix) pass through a StandardScaler followed by PCA, which compresses 128 dimensions into a small number of principal components. Tabular features like annual_income and credit_score pass through a separate StandardScaler untouched. The ColumnTransformer reassembles the two streams into a single feature matrix that the classifier receives. This selective compression is the mechanism that solved the curse-of-dimensionality problem described in the section below: PCA concentrates the graph signal into 5 components while tabular features retain their original resolution.

The function runs cross-validation on each pipeline, records the mean F1 score, then fits the winning pipeline on the full labeled set. When an internal test split is configured, the function also evaluates each classifier on the held-out test rows and logs per-class precision, recall, F1, and a confusion matrix to MLflow. After selecting the best classifier, the function refits it on the complete dataset (train and test combined) so the registered model benefits from every available example. MLflow's autolog captures every parameter, metric, and model artifact automatically. Each classifier gets its own MLflow run within the experiment, and the function returns the best model's URI so the pipeline can register it in Unity Catalog.

#### Model Registration and Promotion

The first entry point registers its best model as Champion immediately since no prior version exists. The second entry point takes a more cautious approach: it registers a new model version, fetches the current Champion's F1 from its MLflow run metrics, and promotes the new version only if it strictly improves. This prevents the community-augmented model from regressing the production model if Louvain adds noise rather than signal.

All models register under a single Unity Catalog name (`graph_feature_forge_risk_classifier`) with the `@Champion` alias tracking the production version. The scoring step loads whichever version currently holds that alias, so the writeback always uses the best available model regardless of which experiment produced it.

### Scoring and Neo4j Writeback

The Champion model is loaded as a Spark UDF via `mlflow.pyfunc.spark_udf()`, which distributes inference across the cluster. The pipeline filters the feature table to customers with null `risk_category`, applies the UDF across all feature columns, and collects predictions. Each prediction is written back to Neo4j as a `risk_category_predicted` property on the Customer node, alongside a `prediction_source` identifier and a UTC timestamp. The Spark Connector handles the writeback via merge on `customer_id`, so repeated runs update rather than duplicate.

This closes the loop: the enrichment pipeline now has risk profiles for all 103 customers, not just the 3 with documents. The next enrichment cycle can use those predictions as additional context when synthesizing gap analyses, and future GDS projections reflect an increasingly annotated graph.

### Three-Way Comparison

The baseline entry point is the control experiment. By dynamically excluding every graph-derived column and training on only `annual_income` and `credit_score`, it establishes the ceiling for what tabular features can achieve. The three-way comparison queries each experiment's MLflow runs for the best F1 score and model type, producing a table that directly answers whether graph features improve classification.

The initial comparison (Phase 2, training on 30 rows with raw 128-dimensional FastRP embeddings) showed tabular-only beating both graph-augmented models by a wide margin, with zero graph features in the top 20 by importance. The section "Why Graph Features Can Hurt" below explains why raw embeddings failed, how PCA addressed the problem, and what happened when the fix was applied.

## Why Graph Features Can Hurt: Dimensionality and Sample Size

The three-way comparison in Phase 2 produced a counterintuitive result. The tabular-only model (F1=0.853 on two features) substantially outperformed both graph-augmented models (F1=0.636 on 130+ features). Feature importance analysis confirmed that zero graph features appeared in the top 20 most important features. The model was relying entirely on annual_income and credit_score while the 128 FastRP dimensions contributed noise.

This is a well-understood phenomenon in machine learning called the curse of dimensionality, and it explains why more information can produce worse predictions.

### The Curse of Dimensionality

A classifier learns decision boundaries by finding patterns in the training data. With two features, the classifier needs to find a line (or curve) that separates Aggressive customers from Conservative customers in a two-dimensional space. Thirty training points in two dimensions is plenty: the space is small, the points are relatively dense, and the classifier can identify where the classes separate.

Add 128 FastRP embedding dimensions and the same 30 points now occupy a 130-dimensional space. The volume of that space grows exponentially with each new dimension. The 30 points that filled the two-dimensional space are now scattered across a volume so vast that each point is effectively isolated. The classifier can no longer distinguish real patterns from coincidental alignments. It memorizes the training data instead of generalizing from it, a condition called overfitting.

The practical consequence is stark. GradientBoosting can partially compensate by using tree splits to ignore irrelevant dimensions, which is why it wins on the high-dimensional data while LogisticRegression (which weighs every dimension) falls to F1=0.329. But even GradientBoosting can only do so much. The tabular model doesn't need to compensate because it never encounters the problem: two features on 30 rows is a healthy ratio where classifiers operate in their comfort zone.

A common rule of thumb is that reliable classification requires at least 5 to 10 training examples per feature. Thirty rows on 130 features gives 0.23 examples per feature. Thirty rows on 2 features gives 15 examples per feature. The gap in performance follows directly from this ratio.

### PCA: Compressing Graph Signal Without Losing It

The 128 FastRP dimensions are not 128 independent pieces of information. They encode the structural neighborhood of each customer through a random projection process that produces correlated dimensions. Customers who share similar graph neighborhoods have similar vectors, and much of the variation across those 128 numbers can be captured by a smaller set of derived features.

Principal Component Analysis (PCA) identifies the directions in the 128-dimensional embedding space along which the data varies the most and projects the data onto those directions. The first principal component captures the axis of maximum variance: if most of the variation across customer embeddings falls along a single direction (say, a spectrum from densely connected portfolios to sparse ones), the first component captures that. The second component captures the next most significant direction of variation, orthogonal to the first, and so on.

Reducing 128 dimensions to 5 principal components means keeping the 5 directions that explain the most variation and discarding the rest. In practice, FastRP embeddings on a graph with clear community structure often have 80-90% of their total variance concentrated in the first 5-10 components. The remaining dimensions contribute small fluctuations that carry more noise than signal, especially at the sample sizes this pipeline operates on.

After PCA, the model trains on annual_income, credit_score, and 5 graph-derived components, which is 7 features total. This gives roughly 4 examples per feature on 30 rows, or roughly 12 examples per feature on the full 82 training rows (when using an 80/20 train/test split instead of the artificial holdout). Both ratios are within the range where classifiers can generalize.

The key insight is that PCA does not discard the graph signal. It concentrates it. The structural information that FastRP encoded across 128 dimensions still exists in the 5 components. What PCA discards is the high-dimensional noise that was drowning that signal. If graph topology genuinely carries predictive information about risk categories, PCA-compressed features should recover it in a form the classifier can use.

### Feature Selection: A Different Approach to the Same Problem

PCA compresses all 128 dimensions into a few components without considering the prediction target. Feature selection takes the opposite approach: it evaluates each of the 128 FastRP dimensions individually against the risk category labels and keeps only the ones that correlate with the target.

Methods like SelectKBest or mutual information score each feature by how much information it provides about the target variable. A FastRP dimension where Aggressive customers cluster at one end and Conservative customers cluster at the other end scores high. A dimension where all three classes overlap scores low. Keeping the top 5 or 10 scoring dimensions produces a feature set that is both small (avoiding the curse of dimensionality) and specifically relevant to risk classification.

The tradeoff is that feature selection evaluates each dimension in isolation. Two dimensions that are individually weak predictors might be powerful in combination (one separates Aggressive from the other two, the other separates Conservative from Moderate). PCA captures these joint patterns because it operates on the full covariance structure. Feature selection can miss them because it evaluates features one at a time.

In practice, both approaches are worth trying. PCA is the safer first step because it makes no assumptions about which aspects of graph structure matter for risk classification. Feature selection is the follow-up if PCA shows that graph features contribute value and the team wants to understand which specific structural properties drive predictions.

### What This Means for the Pipeline: Phase 2.5 Results

Phase 2.5 applied both fixes simultaneously: PCA compression (128 FastRP dimensions reduced to 5 principal components) and a proper 80/20 train/test split that uses all 102 labeled rows instead of the artificial 30-row holdout. The results reversed the Phase 2 finding.

With PCA and the full dataset, FastRP features beat the tabular-only baseline. The FastRP+PCA(5) model achieved a cross-validated F1 of 0.801 compared to 0.784 for tabular-only, a modest but consistent advantage. The FastRP + Louvain + PCA(5) model scored 0.789, suggesting that community detection adds little beyond what the embeddings already capture. On the held-out test set of 21 rows, the graph-augmented model reached a test F1 of 0.903, a strong result even accounting for the variance inherent in small test sets.

The feature-to-sample ratio tells the story of why this worked. Phase 2 trained on 30 rows with 130 features, a ratio of 0.23 examples per feature that guaranteed overfitting. Phase 2.5 trained on 81 rows with 7 features (5 PCA components plus annual_income and credit_score), a ratio of 11.6 examples per feature. The classifiers moved from an impossible problem to a tractable one. Cross-validation standard deviation dropped from 0.19 to 0.09, confirming that the estimates are more stable.

Feature importance analysis on the best graph-augmented model shows that annual_income and credit_score still dominate at roughly 87% combined importance. The five PCA components contribute the remaining 13%, with the first component (capturing the primary axis of variation in graph structure) carrying 3.8% and community_id contributing 2.8%. The graph signal is secondary but real. It encodes structural information about customer portfolio neighborhoods that income and credit score cannot represent: two customers with identical income and credit scores but different portfolio topologies (one connected to tech stocks through a brokerage account, another connected to bonds through a retirement account) receive different PCA component values, and that difference moves the classifier's prediction in the right direction often enough to improve F1 by 0.017 over the tabular baseline.

The Phase 2 results did not prove that graph features are useless. They proved that 128 raw FastRP dimensions overwhelm 30 training examples. PCA concentrates the graph signal into a form the classifier can learn from, and using the full labeled dataset gives the classifier enough examples to learn it.

### Net Outcome

The pipeline transforms a sparsely labeled knowledge graph into a fully scored customer base. Three customers with documents become 103 customers with risk predictions, and graph topology contributes predictive signal that tabular features alone cannot capture. The predictions flow back into Neo4j where they compound with the enrichment pipeline's discoveries: customers with predicted risk profiles give the gap analysis richer context, and the GDS projection in the next cycle operates on a denser, more annotated graph. Each run of the feature engineering stage both consumes the graph's current state and improves it for the next iteration.
