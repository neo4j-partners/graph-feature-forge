# Databricks notebook source
# MAGIC %md
# MAGIC # GDS FastRP Feature Engineering
# MAGIC
# MAGIC Proves the full lifecycle: project the graph-feature-forge portfolio graph in GDS,
# MAGIC compute FastRP embeddings, export to a Delta feature table, train a classifier
# MAGIC with AutoML, score unlabeled customers, and write predictions back to Neo4j.
# MAGIC
# MAGIC **Runtime requirement:** Databricks Runtime 17.x LTS ML or earlier (AutoML removed as built-in in 18.0 ML+).

# COMMAND ----------

# MAGIC %pip install graphdatascience --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os

SECRET_SCOPE = os.environ.get("DATABRICKS_SECRET_SCOPE", "graph-feature-forge")

NEO4J_URI = dbutils.secrets.get(scope=SECRET_SCOPE, key="NEO4J_URI")
NEO4J_USERNAME = dbutils.secrets.get(scope=SECRET_SCOPE, key="NEO4J_USERNAME")
NEO4J_PASSWORD = dbutils.secrets.get(scope=SECRET_SCOPE, key="NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

CATALOG = os.environ.get("CATALOG_NAME", "graph_feature_forge")
SCHEMA = os.environ.get("SCHEMA_NAME", "enrichment")
FEATURE_TABLE = f"`{CATALOG}`.`{SCHEMA}`.customer_graph_features"
ENRICHMENT_LOG_TABLE = f"`{CATALOG}`.`{SCHEMA}`.enrichment_log"

EMBEDDING_DIM = 128
HOLDOUT_PER_CLASS = 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connect to Aura and Project the Graph

# COMMAND ----------

from graphdatascience import GraphDataScience

gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=NEO4J_DATABASE)
print(f"GDS version: {gds.version()}")

# COMMAND ----------

# Discover enrichment relationship types from the enrichment log (if it exists)
enrichment_rel_types = []
try:
    enrichment_rels_df = spark.sql(
        f"SELECT DISTINCT relationship_type FROM {ENRICHMENT_LOG_TABLE}"
    )
    enrichment_rel_types = [row.relationship_type for row in enrichment_rels_df.collect()]
    print(f"Enrichment relationship types found: {enrichment_rel_types}")
except Exception as e:
    print(f"No enrichment log found (first run): {e}")

# COMMAND ----------

# Build relationship spec: base portfolio relationships + any enrichment relationships
relationship_spec = {
    "HAS_ACCOUNT": {"orientation": "UNDIRECTED"},
    "HAS_POSITION": {"orientation": "UNDIRECTED"},
    "OF_SECURITY": {"orientation": "UNDIRECTED"},
    "OF_COMPANY": {"orientation": "UNDIRECTED"},
}

for rel_type in enrichment_rel_types:
    relationship_spec[rel_type] = {"orientation": "UNDIRECTED"}

print(f"Projecting with relationship types: {list(relationship_spec.keys())}")

# COMMAND ----------

# Drop existing projection if present
if gds.graph.exists("portfolio-graph").iloc[0]["exists"]:
    gds.graph.drop(gds.graph.get("portfolio-graph"))
    print("Dropped existing projection")

# Project the portfolio subgraph
G, result = gds.graph.project(
    "portfolio-graph",
    ["Customer", "Account", "Position", "Stock", "Company"],
    relationship_spec,
)

print(f"Projected graph — Nodes: {G.node_count()}, Relationships: {G.relationship_count()}")

# COMMAND ----------

# Verify customer count matches expected 103
customer_count = gds.run_cypher("MATCH (c:Customer) RETURN count(c) AS count").iloc[0]["count"]
print(f"Customer nodes in Neo4j: {customer_count}")
assert customer_count == 103, f"Expected 103 customers, got {customer_count}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute FastRP Embeddings

# COMMAND ----------

result = gds.fastRP.mutate(
    G,
    mutateProperty="fastrp_embedding",
    embeddingDimension=EMBEDDING_DIM,
    iterationWeights=[0.0, 1.0, 1.0, 0.8, 0.5],
    randomSeed=42,
)

print(f"FastRP computed for {result['nodePropertiesWritten']} nodes")

# COMMAND ----------

# Write embeddings back to Neo4j as a Customer node property
gds.graph.nodeProperties.write(G, node_properties=["fastrp_embedding"], node_labels=["Customer"])
print("FastRP embeddings written to Neo4j Customer nodes")

# COMMAND ----------

# Spot-check: verify a customer has the embedding property
check = gds.run_cypher("""
    MATCH (c:Customer)
    WHERE c.fastrp_embedding IS NOT NULL
    RETURN c.customer_id AS id, size(c.fastrp_embedding) AS dim
    LIMIT 3
""")
print("Spot-check (customer_id, embedding dimensions):")
display(check)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Embeddings to a Delta Table

# COMMAND ----------

from pyspark.sql import functions as F

# Read Customer nodes via the Neo4j Spark Connector (same pattern as extraction.py)
customers_df = (
    spark.read.format("org.neo4j.spark.DataSource")
    .option("url", NEO4J_URI)
    .option("authentication.type", "basic")
    .option("authentication.basic.username", NEO4J_USERNAME)
    .option("authentication.basic.password", NEO4J_PASSWORD)
    .option("database", NEO4J_DATABASE)
    .option("labels", ":Customer")
    .load()
)

print(f"Columns from Neo4j: {customers_df.columns}")
customers_df.printSchema()

# COMMAND ----------

# Build the feature DataFrame: tabular features + exploded embeddings
feature_df = customers_df.select(
    F.col("customer_id"),
    F.col("annual_income").cast("double"),
    F.col("credit_score").cast("double"),
    F.col("risk_profile").alias("risk_category"),
    F.col("fastrp_embedding"),
)

# Explode the 128-dim embedding into individual columns (fastrp_0 .. fastrp_127)
for i in range(EMBEDDING_DIM):
    feature_df = feature_df.withColumn(
        f"fastrp_{i}", F.col("fastrp_embedding").getItem(i).cast("double")
    )

feature_df = feature_df.drop("fastrp_embedding")

print(f"Feature columns: {len(feature_df.columns)}")
display(feature_df.limit(5))

# COMMAND ----------

# Write to Unity Catalog as a managed Delta table
feature_df.write.mode("overwrite").saveAsTable(FEATURE_TABLE)

# Count from the written table to avoid re-evaluating the Spark plan
row_count = spark.table(FEATURE_TABLE).count()
print(f"Feature table written to {FEATURE_TABLE} ({row_count} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate the Prediction Scenario
# MAGIC
# MAGIC Hold out most labels to simulate the real use case: only a few customers have
# MAGIC known risk profiles, and the classifier predicts the rest.

# COMMAND ----------

import pandas as pd
import numpy as np

# Read the feature table back
features_pdf = spark.table(FEATURE_TABLE).toPandas()

# Check the label distribution
label_counts = features_pdf["risk_category"].value_counts(dropna=False)
print("Label distribution before holdout:")
print(label_counts)

# COMMAND ----------

# Stratified sample: keep HOLDOUT_PER_CLASS labels per category, null the rest
np.random.seed(42)

labeled_mask = features_pdf["risk_category"].notna() & (features_pdf["risk_category"] != "")
labeled_df = features_pdf[labeled_mask]

keep_indices = []
for category in labeled_df["risk_category"].unique():
    category_indices = labeled_df[labeled_df["risk_category"] == category].index.tolist()
    keep = np.random.choice(category_indices, size=min(HOLDOUT_PER_CLASS, len(category_indices)), replace=False)
    keep_indices.extend(keep)

# Record ground truth before nulling
ground_truth = features_pdf[["customer_id", "risk_category"]].copy()
ground_truth.columns = ["customer_id", "true_risk_category"]

# Null out held-out labels
holdout_mask = labeled_mask & ~features_pdf.index.isin(keep_indices)
features_pdf.loc[holdout_mask, "risk_category"] = None

kept_count = features_pdf["risk_category"].notna().sum()
held_out_count = holdout_mask.sum()
print(f"Labels kept for training: {kept_count}")
print(f"Labels held out (to predict): {held_out_count}")
print(f"\nLabel distribution after holdout:")
print(features_pdf["risk_category"].value_counts(dropna=False))

# COMMAND ----------

# Write the holdout feature table back to Delta (overwrite)
holdout_sdf = spark.createDataFrame(features_pdf)
holdout_sdf.write.mode("overwrite").saveAsTable(FEATURE_TABLE)
print(f"Holdout feature table written to {FEATURE_TABLE}")

# Save ground truth for later comparison
ground_truth_table = f"`{CATALOG}`.`{SCHEMA}`.holdout_ground_truth"
spark.createDataFrame(ground_truth).write.mode("overwrite").saveAsTable(ground_truth_table)
print(f"Ground truth saved to {ground_truth_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train with Databricks AutoML

# COMMAND ----------

from databricks import automl

summary = automl.classify(
    dataset=FEATURE_TABLE,
    target_col="risk_category",
    primary_metric="f1",
    exclude_cols=["customer_id"],
    timeout_minutes=30,
    experiment_name="/Shared/graph-feature-forge/fastrp_risk_classification",
)

# COMMAND ----------

print(f"Best trial metric (F1): {summary.best_trial.evaluation_metric_score:.4f}")
print(f"Best model: {summary.best_trial.model_description}")
print(f"Number of trials: {len(summary.trials)}")
print(f"Generated notebook: {summary.best_trial.notebook_path}")

# COMMAND ----------

# Print the leaderboard
for i, trial in enumerate(
    sorted(summary.trials, key=lambda t: t.evaluation_metric_score, reverse=True)
):
    print(
        f"#{i+1} | {trial.model_description:40s} | "
        f"F1={trial.evaluation_metric_score:.4f}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the Best Model with Champion Alias

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

model_name = f"{CATALOG}.{SCHEMA}.graph_feature_forge_risk_classifier"
model_uri = summary.best_trial.model_path

registered_model = mlflow.register_model(model_uri, model_name)
print(f"Registered model: {registered_model.name}, version {registered_model.version}")

# Set the Champion alias so the scoring step can load by alias
client = mlflow.MlflowClient()
client.set_registered_model_alias(model_name, "Champion", registered_model.version)
print(f"Set 'Champion' alias to version {registered_model.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score Unlabeled Customers and Write Back

# COMMAND ----------

# Load the feature table and filter to held-out customers (null risk_category)
features_df = spark.table(FEATURE_TABLE)
unlabeled_df = features_df.filter(F.col("risk_category").isNull())
unlabeled_count = unlabeled_df.count()
print(f"Customers to score: {unlabeled_count}")

# COMMAND ----------

# Score using the Champion model
predict_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{model_name}@Champion",
    result_type="string",
)

feature_cols = [c for c in unlabeled_df.columns if c not in ["risk_category", "customer_id"]]
predictions_df = unlabeled_df.withColumn(
    "predicted_risk_category", predict_udf(*[F.col(c) for c in feature_cols])
)

display(predictions_df.select("customer_id", "predicted_risk_category"))

# COMMAND ----------

# Compare predictions against ground truth
ground_truth_df = spark.table(ground_truth_table)

comparison_df = (
    predictions_df.select("customer_id", "predicted_risk_category")
    .join(ground_truth_df, on="customer_id")
    .withColumn(
        "correct",
        F.col("predicted_risk_category") == F.col("true_risk_category"),
    )
)

total = comparison_df.count()
correct = comparison_df.filter(F.col("correct")).count()
accuracy = correct / total if total > 0 else 0

print(f"Accuracy on held-out customers: {correct}/{total} = {accuracy:.2%}")
print(f"Baseline (majority class): ~36%")

display(comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Predictions Back to Neo4j

# COMMAND ----------

writeback_df = predictions_df.select(
    F.col("customer_id"),
    F.col("predicted_risk_category").alias("risk_category_predicted"),
    F.lit("automl_fastrp").alias("prediction_source"),
    F.current_timestamp().alias("prediction_timestamp"),
)

(
    writeback_df.write.format("org.neo4j.spark.DataSource")
    .option("url", NEO4J_URI)
    .option("authentication.type", "basic")
    .option("authentication.basic.username", NEO4J_USERNAME)
    .option("authentication.basic.password", NEO4J_PASSWORD)
    .option("database", NEO4J_DATABASE)
    .option("labels", ":Customer")
    .option("node.keys", "customer_id")
    .mode("Overwrite")
    .save()
)

print(f"Wrote predictions for {unlabeled_count} customers back to Neo4j")

# COMMAND ----------

# Verify in Neo4j
verify = gds.run_cypher("""
    MATCH (c:Customer)
    WHERE c.risk_category_predicted IS NOT NULL
    RETURN c.customer_id AS id, c.risk_category_predicted AS predicted, c.prediction_source AS source
    LIMIT 5
""")
print("Spot-check predictions in Neo4j:")
display(verify)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the Enrichment Pipeline
# MAGIC
# MAGIC Run the existing graph-feature-forge enrichment pipeline against the graph that now
# MAGIC has predicted risk profiles on every customer. Compare enrichment results
# MAGIC against a baseline run without the predictions.
# MAGIC
# MAGIC This step is run separately via the graph-feature-forge pipeline orchestrator:
# MAGIC ```
# MAGIC python agent_modules/run_graph_feature_forge.py --execute
# MAGIC ```
# MAGIC
# MAGIC The comparison answers whether the LLM synthesis step produces more or
# MAGIC different proposals when every customer has a risk profile versus when only
# MAGIC the original document-backed customers have rich context.

# COMMAND ----------

# Clean up the GDS projection
G.drop()
print("GDS projection dropped")
