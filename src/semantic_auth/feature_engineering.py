"""GDS feature engineering for the semantic-auth enrichment pipeline.

Connects to a Neo4j Aura instance with the GDS plugin enabled, projects
the portfolio graph, computes FastRP embeddings and Louvain community IDs,
exports features to a Delta table, and scores unlabeled customers using
a registered MLflow model.

This module is opt-in: the pipeline orchestrator calls these functions
only when ``ENABLE_GDS_FEATURES=true``.
"""

from __future__ import annotations

from typing import Any

from semantic_auth.extraction import _spark_neo4j_options


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 128
PROJECTION_NAME = "portfolio-graph"
FEATURE_TABLE_NAME = "customer_graph_features"

_BASE_RELATIONSHIP_SPEC = {
    "HAS_ACCOUNT": {"orientation": "UNDIRECTED"},
    "HAS_POSITION": {"orientation": "UNDIRECTED"},
    "OF_SECURITY": {"orientation": "UNDIRECTED"},
    "OF_COMPANY": {"orientation": "UNDIRECTED"},
}

_NODE_LABELS = ["Customer", "Account", "Position", "Stock", "Company"]


# ---------------------------------------------------------------------------
# GDS feature computation
# ---------------------------------------------------------------------------


def compute_gds_features(
    uri: str,
    username: str,
    password: str,
    database: str,
    enrichment_rel_types: list[str] | None = None,
) -> dict[str, Any]:
    """Project the portfolio graph in GDS, run FastRP and Louvain,
    and write computed properties back to Neo4j Customer nodes.

    Returns a stats dict with node_count, community_count, and modularity.
    """
    from graphdatascience import GraphDataScience

    gds = GraphDataScience(uri, auth=(username, password), database=database)

    # Build relationship spec: base + enrichment
    relationship_spec = dict(_BASE_RELATIONSHIP_SPEC)
    for rel_type in enrichment_rel_types or []:
        relationship_spec[rel_type] = {"orientation": "UNDIRECTED"}

    # Drop existing projection if present
    if gds.graph.exists(PROJECTION_NAME).iloc[0]["exists"]:
        gds.graph.drop(gds.graph.get(PROJECTION_NAME))

    G, _ = gds.graph.project(PROJECTION_NAME, _NODE_LABELS, relationship_spec)
    print(f"    Projected graph — Nodes: {G.node_count()}, Relationships: {G.relationship_count()}")

    # FastRP embeddings
    fastrp_result = gds.fastRP.mutate(
        G,
        mutateProperty="fastrp_embedding",
        embeddingDimension=EMBEDDING_DIM,
        iterationWeights=[0.0, 1.0, 1.0, 0.8, 0.5],
        randomSeed=42,
    )
    print(f"    FastRP: {fastrp_result['nodePropertiesWritten']} nodes")

    # Louvain community detection
    louvain_result = gds.louvain.mutate(
        G,
        mutateProperty="community_id",
        maxLevels=10,
        maxIterations=10,
    )
    print(
        f"    Louvain: {louvain_result['communityCount']} communities, "
        f"modularity {louvain_result['modularity']:.4f}"
    )

    # Write properties back to Neo4j
    gds.graph.nodeProperties.write(
        G, node_properties=["fastrp_embedding"], node_labels=["Customer"],
    )
    gds.graph.nodeProperties.write(
        G, node_properties=["community_id"], node_labels=["Customer"],
    )
    print("    Properties written to Neo4j Customer nodes")

    # Clean up
    G.drop()

    return {
        "node_count": G.node_count(),
        "relationship_count": G.relationship_count(),
        "community_count": louvain_result["communityCount"],
        "modularity": louvain_result["modularity"],
    }


# ---------------------------------------------------------------------------
# Feature table export
# ---------------------------------------------------------------------------


def export_feature_table(
    spark: Any,
    uri: str,
    username: str,
    password: str,
    database: str,
    catalog: str,
    schema: str,
) -> int:
    """Read Customer nodes from Neo4j via the Spark Connector, explode
    FastRP embeddings into individual columns, and write a Delta feature
    table to Unity Catalog.

    Returns the number of rows written.
    """
    from pyspark.sql import functions as F

    options = _spark_neo4j_options(uri, username, password, database)
    options["labels"] = ":Customer"

    customers_df = (
        spark.read.format("org.neo4j.spark.DataSource")
        .options(**options)
        .load()
    )

    # Build feature DataFrame: tabular + community_id + exploded embeddings
    feature_df = customers_df.select(
        F.col("customer_id"),
        F.col("annual_income").cast("double"),
        F.col("credit_score").cast("double"),
        F.col("risk_profile").alias("risk_category"),
        F.col("community_id").cast("string"),  # categorical
        F.col("fastrp_embedding"),
    )

    for i in range(EMBEDDING_DIM):
        feature_df = feature_df.withColumn(
            f"fastrp_{i}", F.col("fastrp_embedding").getItem(i).cast("double"),
        )
    feature_df = feature_df.drop("fastrp_embedding")

    table_name = f"`{catalog}`.`{schema}`.`{FEATURE_TABLE_NAME}`"
    feature_df.write.mode("overwrite").saveAsTable(table_name)

    count = feature_df.count()
    print(f"    Feature table: {table_name} ({count} rows, {len(feature_df.columns)} columns)")
    return count


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_unlabeled_customers(
    spark: Any,
    uri: str,
    username: str,
    password: str,
    database: str,
    catalog: str,
    schema: str,
    model_name: str,
) -> int:
    """Load the Champion model from MLflow, score customers with null
    risk_category, and write predictions back to Neo4j.

    Skips gracefully if no registered model exists.
    Returns the number of customers scored.
    """
    import mlflow
    from pyspark.sql import functions as F

    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()

    # Check if the model exists
    try:
        client.get_model_version_by_alias(model_name, "Champion")
    except mlflow.exceptions.MlflowException:
        print("    No registered Champion model found — skipping scoring")
        return 0

    # Load feature table
    table_name = f"`{catalog}`.`{schema}`.`{FEATURE_TABLE_NAME}`"
    features_df = spark.table(table_name)
    unlabeled_df = features_df.filter(F.col("risk_category").isNull())

    unlabeled_count = unlabeled_df.count()
    if unlabeled_count == 0:
        print("    No unlabeled customers to score")
        return 0

    # Score with the Champion model
    predict_udf = mlflow.pyfunc.spark_udf(
        spark,
        model_uri=f"models:/{model_name}@Champion",
        result_type="string",
    )

    feature_cols = [
        c for c in unlabeled_df.columns if c not in ("risk_category", "customer_id")
    ]
    predictions_df = unlabeled_df.withColumn(
        "predicted_risk_category",
        predict_udf(*[F.col(c) for c in feature_cols]),
    )

    # Write predictions back to Neo4j
    writeback_df = predictions_df.select(
        F.col("customer_id"),
        F.col("predicted_risk_category").alias("risk_category_predicted"),
        F.lit("automl_gds_pipeline").alias("prediction_source"),
        F.current_timestamp().alias("prediction_timestamp"),
    )

    options = _spark_neo4j_options(uri, username, password, database)
    options["labels"] = ":Customer"
    options["node.keys"] = "customer_id"

    (
        writeback_df.write.format("org.neo4j.spark.DataSource")
        .options(**options)
        .mode("Overwrite")
        .save()
    )

    scored = writeback_df.count()
    print(f"    Scored {scored} customers, predictions written to Neo4j")
    return scored
