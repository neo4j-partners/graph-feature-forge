# GDS Feature Engineering Notebooks

Standalone Databricks notebooks that compute graph features from the graph_feature_forge portfolio graph and train classifiers with AutoML. These notebooks run independently of the enrichment pipeline and are designed to be executed in sequence on Databricks Runtime 17.x LTS ML.


## Notebooks

| Notebook | What it does | Depends on |
|----------|-------------|------------|
| `gds_fastrp_features.py` | Projects the portfolio graph in GDS, computes 128-dim FastRP embeddings, exports to a Delta feature table, creates a stratified holdout, trains a classifier with AutoML, scores held-out customers, and writes predictions back to Neo4j. Registers the model with a Champion alias in Unity Catalog. | Neo4j Aura with GDS enabled |
| `gds_community_features.py` | Adds Louvain community detection as a categorical feature alongside FastRP. Retrains AutoML with the combined feature set and promotes the Champion model if F1 improves. Runs kNN to visualize nearest-neighbor relationships and community overlap. | `gds_fastrp_features` (holdout split, ground truth table) |
| `gds_baseline_comparison.py` | Trains a tabular-only model (annual_income, credit_score) excluding all graph features. Produces a three-way MLflow comparison table and feature importance analysis showing whether graph features carry signal beyond tabular attributes. | `gds_fastrp_features` (holdout split, feature table) |

## Prerequisites

- Databricks Runtime **17.x LTS ML** (AutoML removed as built-in in 18.0 ML+)
- Neo4j Aura instance with the GDS plugin enabled
- Neo4j Spark Connector (`org.neo4j:neo4j-connector-apache-spark_2.12:5.x`) configured as a cluster library
- Environment variables: `NEO4J_URI`, `NEO4J_PASSWORD`, `CATALOG_NAME`, `SCHEMA_NAME`

## MLflow Experiments

Each notebook writes to a separate MLflow experiment for side-by-side comparison:

| Experiment | Features used |
|------------|--------------|
| `/Shared/graph_feature_forge/fastrp_risk_classification` | 128 FastRP dimensions + tabular |
| `/Shared/graph_feature_forge/fastrp_louvain_risk_classification` | 128 FastRP dimensions + community_id + tabular |
| `/Shared/graph_feature_forge/tabular_only_baseline` | Tabular only (annual_income, credit_score) |
