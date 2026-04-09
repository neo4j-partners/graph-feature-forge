# Databricks notebook source
# MAGIC %md
# MAGIC # GDS Community Detection Features
# MAGIC
# MAGIC Adds Louvain community detection to the existing FastRP workflow. Produces a
# MAGIC single additional categorical feature column (community_id), retrains AutoML
# MAGIC with the combined feature set, and compares against the FastRP-only result.
# MAGIC
# MAGIC **Prerequisite:** Run `gds_fastrp_features` first. This notebook reuses the
# MAGIC holdout split and ground truth table created there.
# MAGIC
# MAGIC **Runtime requirement:** Databricks Runtime 17.x LTS ML or earlier.

# COMMAND ----------

# MAGIC %pip install graphdatascience --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os

SECRET_SCOPE = os.environ.get("DATABRICKS_SECRET_SCOPE", "graph_feature_forge")

NEO4J_URI = dbutils.secrets.get(scope=SECRET_SCOPE, key="NEO4J_URI")
NEO4J_USERNAME = dbutils.secrets.get(scope=SECRET_SCOPE, key="NEO4J_USERNAME")
NEO4J_PASSWORD = dbutils.secrets.get(scope=SECRET_SCOPE, key="NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

CATALOG = os.environ.get("CATALOG_NAME", "graph_feature_forge")
SCHEMA = os.environ.get("SCHEMA_NAME", "enrichment")
FEATURE_TABLE = f"`{CATALOG}`.`{SCHEMA}`.customer_graph_features"
ENRICHMENT_LOG_TABLE = f"`{CATALOG}`.`{SCHEMA}`.enrichment_log"

EMBEDDING_DIM = 128

# COMMAND ----------

# MAGIC %md
# MAGIC ## Project the Graph and Compute FastRP + Louvain

# COMMAND ----------

from graphdatascience import GraphDataScience

gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=NEO4J_DATABASE)
print(f"GDS version: {gds.version()}")

# COMMAND ----------

# Discover enrichment relationship types
enrichment_rel_types = []
try:
    enrichment_rels_df = spark.sql(
        f"SELECT DISTINCT relationship_type FROM {ENRICHMENT_LOG_TABLE}"
    )
    enrichment_rel_types = [row.relationship_type for row in enrichment_rels_df.collect()]
    print(f"Enrichment relationship types found: {enrichment_rel_types}")
except Exception:
    print("No enrichment log found")

# COMMAND ----------

# Build relationship spec
relationship_spec = {
    "HAS_ACCOUNT": {"orientation": "UNDIRECTED"},
    "HAS_POSITION": {"orientation": "UNDIRECTED"},
    "OF_SECURITY": {"orientation": "UNDIRECTED"},
    "OF_COMPANY": {"orientation": "UNDIRECTED"},
}

for rel_type in enrichment_rel_types:
    relationship_spec[rel_type] = {"orientation": "UNDIRECTED"}

# Drop existing projection if present
try:
    gds.graph.drop(gds.graph.get("portfolio-graph"))
except Exception:
    pass

G, result = gds.graph.project(
    "portfolio-graph",
    ["Customer", "Account", "Position", "Stock", "Company"],
    relationship_spec,
)

print(f"Projected graph — Nodes: {G.node_count()}, Relationships: {G.relationship_count()}")

# COMMAND ----------

# Recompute FastRP (needed for the combined feature table)
fastrp_result = gds.fastRP.mutate(
    G,
    mutateProperty="fastrp_embedding",
    embeddingDimension=EMBEDDING_DIM,
    iterationWeights=[0.0, 1.0, 1.0, 0.8, 0.5],
    randomSeed=42,
)
print(f"FastRP computed for {fastrp_result['nodePropertiesWritten']} nodes")

# COMMAND ----------

# Run Louvain community detection
louvain_result = gds.louvain.mutate(
    G,
    mutateProperty="community_id",
    maxLevels=10,
    maxIterations=10,
)

print(f"Communities found: {louvain_result['communityCount']}")
print(f"Modularity: {louvain_result['modularity']:.4f}")

# COMMAND ----------

# Write both properties back to Neo4j
gds.graph.nodeProperties.write(G, node_properties=["fastrp_embedding"], node_labels=["Customer"])
gds.graph.nodeProperties.write(G, node_properties=["community_id"], node_labels=["Customer"])
print("FastRP embeddings and community IDs written to Neo4j")

# COMMAND ----------

# Inspect communities: which customers share a community?
communities = gds.run_cypher("""
    MATCH (c:Customer)
    WHERE c.community_id IS NOT NULL
    RETURN c.community_id AS community, count(c) AS members,
           collect(c.first_name + ' ' + c.last_name)[..5] AS sample_names
    ORDER BY members DESC
""")
print("Community distribution:")
display(communities)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Community ID to the Feature Table

# COMMAND ----------

from pyspark.sql import functions as F

# Re-export Customer nodes with both FastRP and community_id
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

# COMMAND ----------

# Build feature table: tabular + community_id (categorical) + exploded FastRP
from graph_feature_forge.ml.feature_engineering import parse_and_explode_embedding

feature_df = customers_df.select(
    F.col("customer_id"),
    F.col("annual_income").cast("double"),
    F.col("credit_score").cast("double"),
    F.col("risk_profile").alias("risk_category"),
    F.col("community_id").cast("string"),  # categorical, not numeric
    F.col("fastrp_embedding"),
)

# Handles both native array and JSON string formats from the Neo4j Spark Connector
feature_df = parse_and_explode_embedding(feature_df, embedding_dim=EMBEDDING_DIM)

print(f"Feature table: {feature_df.count()} rows, {len(feature_df.columns)} columns")
print(f"New column: community_id (categorical)")

# COMMAND ----------

# Re-apply the same holdout from gds_fastrp_features
ground_truth_table = f"`{CATALOG}`.`{SCHEMA}`.holdout_ground_truth"

import pandas as pd
import numpy as np

features_pdf = feature_df.toPandas()

# Re-read the existing feature table's null pattern to match the original holdout
phase1_features = spark.table(FEATURE_TABLE).select("customer_id", "risk_category").toPandas()
nulled_ids = set(phase1_features[phase1_features["risk_category"].isna()]["customer_id"])

features_pdf.loc[features_pdf["customer_id"].isin(nulled_ids), "risk_category"] = None

kept = features_pdf["risk_category"].notna().sum()
held_out = features_pdf["risk_category"].isna().sum()
print(f"Labels kept: {kept}, held out: {held_out} (same split as gds_fastrp_features)")

# COMMAND ----------

# Overwrite the feature table with the new columns
holdout_sdf = spark.createDataFrame(features_pdf)
holdout_sdf.write.mode("overwrite").saveAsTable(FEATURE_TABLE)
print(f"Updated feature table written to {FEATURE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrain AutoML with Combined Feature Set

# COMMAND ----------

from databricks import automl

summary = automl.classify(
    dataset=FEATURE_TABLE,
    target_col="risk_category",
    primary_metric="f1",
    exclude_cols=["customer_id"],
    timeout_minutes=30,
    experiment_name="/Shared/graph_feature_forge/fastrp_louvain_risk_classification",
)

# COMMAND ----------

print(f"Best trial metric (F1): {summary.best_trial.evaluation_metric_score:.4f}")
print(f"Best model: {summary.best_trial.model_description}")
print(f"Number of trials: {len(summary.trials)}")

# COMMAND ----------

# Check if community_id contributes to the model
print(f"\nCompare in MLflow:")
print(f"  FastRP only:      /Shared/graph_feature_forge/fastrp_risk_classification")
print(f"  FastRP + Louvain: /Shared/graph_feature_forge/fastrp_louvain_risk_classification")
print(f"\nOpen the MLflow UI to compare best runs side by side.")

# COMMAND ----------

# Update the Champion model if this run outperforms the current Champion
import mlflow

mlflow.set_registry_uri("databricks-uc")

model_name = f"{CATALOG}.{SCHEMA}.graph_feature_forge_risk_classifier"
model_uri = summary.best_trial.model_path

registered_model = mlflow.register_model(model_uri, model_name)
print(f"Registered version {registered_model.version}")

client = mlflow.MlflowClient()

# Compare against current Champion
current_champion = client.get_model_version_by_alias(model_name, "Champion")
champion_run = client.get_run(current_champion.run_id)
champion_f1 = champion_run.data.metrics.get("val_f1_score", 0)
new_f1 = summary.best_trial.evaluation_metric_score

print(f"Current Champion F1: {champion_f1:.4f}")
print(f"FastRP + Louvain F1: {new_f1:.4f}")

if new_f1 > champion_f1:
    client.set_registered_model_alias(model_name, "Champion", registered_model.version)
    print(f"Promoted version {registered_model.version} to Champion")
else:
    print("Current Champion retained (FastRP + Louvain did not improve)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Communities with kNN
# MAGIC
# MAGIC Run kNN on the FastRP embeddings to find each customer's nearest neighbors.
# MAGIC Shows that the embeddings capture meaningful investment structure.

# COMMAND ----------

# Run kNN on the in-memory projection using FastRP embeddings
knn_result = gds.knn.mutate(
    G,
    mutateRelationshipType="SIMILAR_TO",
    mutateProperty="similarity",
    nodeProperties=["fastrp_embedding"],
    topK=5,
    randomSeed=42,
    sampleRate=1.0,
    deltaThreshold=0.001,
)

print(f"kNN relationships created: {knn_result['relationshipsWritten']}")

# COMMAND ----------

# Show nearest neighbors for specific customers
spotlight_customers = ["James Anderson", "Maria Rodriguez", "Robert Chen"]

for name in spotlight_customers:
    parts = name.split()
    first, last = parts[0], parts[1]

    neighbors = gds.run_cypher(f"""
        MATCH (c:Customer {{first_name: '{first}', last_name: '{last}'}})
        MATCH (c)-[r:SIMILAR_TO]->(neighbor:Customer)
        RETURN c.first_name + ' ' + c.last_name AS customer,
               c.community_id AS community,
               neighbor.first_name + ' ' + neighbor.last_name AS neighbor_name,
               neighbor.community_id AS neighbor_community,
               r.similarity AS similarity
        ORDER BY r.similarity DESC
        LIMIT 5
    """)

    if not neighbors.empty:
        print(f"\n{name} (community {neighbors.iloc[0]['community']}):")
        display(neighbors)
    else:
        print(f"\n{name}: not found or no neighbors")

# COMMAND ----------

# Community overlap summary: do nearest neighbors tend to be in the same community?
overlap = gds.run_cypher("""
    MATCH (c:Customer)-[r:SIMILAR_TO]->(n:Customer)
    RETURN
        CASE WHEN c.community_id = n.community_id THEN 'same' ELSE 'different' END AS community_match,
        count(*) AS count,
        avg(r.similarity) AS avg_similarity
""")
print("kNN neighbor community overlap:")
display(overlap)

# COMMAND ----------

# Clean up
G.drop()
print("GDS projection dropped")
