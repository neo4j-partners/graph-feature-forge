"""GDS FastRP feature engineering: project → FastRP → export → holdout → AutoML → score → writeback.

Proves the full lifecycle: project the portfolio graph in GDS, compute FastRP
embeddings, export to a Delta feature table, train a classifier with AutoML,
score unlabeled customers, and write predictions back to Neo4j.

Usage:
    python -m cli upload --wheel
    python -m cli upload gds_fastrp_features.py
    python -m cli submit gds_fastrp_features.py --compute cluster
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GDSFastRPConfig:
    """Configuration from environment variables and CLI flags."""

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    catalog: str
    schema: str
    embedding_dim: int
    holdout_per_class: int
    test_size: float | None
    pca_components: int | None

    @property
    def feature_table(self) -> str:
        return f"{self.catalog}.{self.schema}.customer_graph_features"

    @property
    def enrichment_log_table(self) -> str:
        return f"{self.catalog}.{self.schema}.enrichment_log"

    @property
    def ground_truth_table(self) -> str:
        return f"{self.catalog}.{self.schema}.holdout_ground_truth"

    @property
    def model_name(self) -> str:
        return f"{self.catalog}.{self.schema}.graph_feature_forge_risk_classifier"

    @classmethod
    def from_env(cls) -> GDSFastRPConfig:
        from graph_feature_forge import inject_params

        inject_params()

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--embedding-dim", type=int, default=128)
        parser.add_argument("--holdout-per-class", type=int, default=10)
        flags, _ = parser.parse_known_args()

        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        if not all([neo4j_uri, neo4j_username, neo4j_password]):
            print("ERROR: Neo4j connection required.")
            print("  Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env")
            sys.exit(1)

        test_size_str = os.getenv("TEST_SIZE")
        test_size = float(test_size_str) if test_size_str else None

        pca_str = os.getenv("PCA_COMPONENTS")
        pca_components = int(pca_str) if pca_str else None

        return cls(
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            catalog=os.getenv("CATALOG_NAME", "graph_feature_forge"),
            schema=os.getenv("SCHEMA_NAME", "enrichment"),
            embedding_dim=flags.embedding_dim,
            holdout_per_class=flags.holdout_per_class,
            test_size=test_size,
            pca_components=pca_components,
        )


# ---------------------------------------------------------------------------
# Authentication (same pattern as run_graph_feature_forge.py)
# ---------------------------------------------------------------------------


def _authenticate() -> Any:
    """Connect to Databricks and return the WorkspaceClient."""
    from databricks.sdk import WorkspaceClient

    wc = WorkspaceClient()
    print(f"  Connected to {wc.config.host}")
    return wc


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _discover_enrichment_rels(spark: Any, enrichment_log_table: str) -> list[str]:
    """Read distinct relationship types from the enrichment log."""
    try:
        rows = spark.sql(
            f"SELECT DISTINCT relationship_type FROM {enrichment_log_table}"
        ).collect()
        rel_types = [row.relationship_type for row in rows]
        print(f"  Enrichment relationship types: {rel_types}")
        return rel_types
    except Exception as e:
        print(f"  No enrichment log found (first run): {e}")
        return []


def _compute_fastrp(cfg: GDSFastRPConfig, enrichment_rel_types: list[str]) -> None:
    """Project the graph in GDS, compute FastRP, and write to Neo4j."""
    from graph_feature_forge.ml.feature_engineering import compute_gds_features

    # compute_gds_features runs FastRP + Louvain; we use both here
    # even though this is the "FastRP-only" notebook — Louvain is written
    # to Neo4j but excluded from the feature table in this step.
    stats = compute_gds_features(
        uri=cfg.neo4j_uri,
        username=cfg.neo4j_username,
        password=cfg.neo4j_password,
        database=cfg.neo4j_database,
        enrichment_rel_types=enrichment_rel_types,
    )
    print(f"  Nodes: {stats['node_count']}, Relationships: {stats['relationship_count']}")
    print(f"  Communities: {stats['community_count']}, Modularity: {stats['modularity']:.4f}")


def _export_features(spark: Any, cfg: GDSFastRPConfig) -> None:
    """Export FastRP embeddings (without community_id) to a Delta feature table."""
    from pyspark.sql import functions as F

    from graph_feature_forge.graph.extraction import spark_neo4j_options

    options = spark_neo4j_options(
        cfg.neo4j_uri, cfg.neo4j_username, cfg.neo4j_password, cfg.neo4j_database,
    )
    options["labels"] = ":Customer"

    customers_df = (
        spark.read.format("org.neo4j.spark.DataSource")
        .options(**options)
        .load()
    )

    # FastRP-only feature table (no community_id — that's the community notebook)
    from graph_feature_forge.ml.feature_engineering import parse_and_explode_embedding

    feature_df = customers_df.select(
        F.col("customer_id"),
        F.col("annual_income").cast("double"),
        F.col("credit_score").cast("double"),
        F.col("risk_profile").alias("risk_category"),
        F.col("fastrp_embedding"),
    )

    feature_df = parse_and_explode_embedding(
        feature_df, embedding_dim=cfg.embedding_dim,
    )

    feature_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.feature_table)
    count = spark.table(cfg.feature_table).count()
    print(f"  Feature table: {cfg.feature_table} ({count} rows, {len(feature_df.columns)} cols)")


def _create_holdout(spark: Any, cfg: GDSFastRPConfig) -> None:
    """Stratified holdout simulation — null most labels, save ground truth."""
    from graph_feature_forge.ml.automl_training import create_holdout

    features_pdf = spark.table(cfg.feature_table).toPandas()
    holdout_pdf, ground_truth_pdf = create_holdout(
        features_pdf,
        holdout_per_class=cfg.holdout_per_class,
    )

    spark.createDataFrame(holdout_pdf).write.mode("overwrite").saveAsTable(cfg.feature_table)
    spark.createDataFrame(ground_truth_pdf).write.mode("overwrite").saveAsTable(
        cfg.ground_truth_table,
    )
    print(f"  Ground truth saved to {cfg.ground_truth_table}")


def _train_and_register(cfg: GDSFastRPConfig) -> Any:
    """Train sklearn classifier and register as Champion."""
    from graph_feature_forge.ml.automl_training import register_model, train_sklearn_classifier

    summary = train_sklearn_classifier(
        feature_table=cfg.feature_table,
        experiment_name="/Shared/graph-feature-forge/fastrp_risk_classification",
        test_size=cfg.test_size,
        pca_components=cfg.pca_components,
    )

    register_model(
        model_uri=summary.best_trial.model_path,
        model_name=cfg.model_name,
        set_champion=True,
    )

    return summary


def _score_and_evaluate(spark: Any, cfg: GDSFastRPConfig) -> None:
    """Score held-out customers and evaluate against ground truth."""
    import mlflow
    from pyspark.sql import functions as F

    from graph_feature_forge.ml.automl_training import evaluate_predictions
    from graph_feature_forge.graph.extraction import spark_neo4j_options

    mlflow.set_registry_uri("databricks-uc")

    features_df = spark.table(cfg.feature_table)
    unlabeled_df = features_df.filter(F.col("risk_category").isNull())
    unlabeled_count = unlabeled_df.count()
    print(f"  Customers to score: {unlabeled_count}")

    predict_udf = mlflow.pyfunc.spark_udf(
        spark,
        model_uri=f"models:/{cfg.model_name}@Champion",
        result_type="string",
    )

    feature_cols = [c for c in unlabeled_df.columns if c not in ("risk_category", "customer_id")]
    predictions_df = unlabeled_df.withColumn(
        "predicted_risk_category",
        predict_udf(*[F.col(c) for c in feature_cols]),
    )

    evaluate_predictions(spark, predictions_df, cfg.ground_truth_table)

    # Write predictions back to Neo4j
    writeback_df = predictions_df.select(
        F.col("customer_id"),
        F.col("predicted_risk_category").alias("risk_category_predicted"),
        F.lit("automl_fastrp").alias("prediction_source"),
        F.current_timestamp().alias("prediction_timestamp"),
    )

    options = spark_neo4j_options(
        cfg.neo4j_uri, cfg.neo4j_username, cfg.neo4j_password, cfg.neo4j_database,
    )
    options["labels"] = ":Customer"
    options["node.keys"] = "customer_id"

    (
        writeback_df.write.format("org.neo4j.spark.DataSource")
        .options(**options)
        .mode("Overwrite")
        .save()
    )

    print(f"  Predictions written to Neo4j for {unlabeled_count} customers")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = GDSFastRPConfig.from_env()

    print("=" * 60)
    print("GDS FastRP Feature Engineering")
    print("=" * 60)

    _authenticate()

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    print("\nStep 1/5: Computing GDS features ...")
    enrichment_rels = _discover_enrichment_rels(spark, cfg.enrichment_log_table)
    _compute_fastrp(cfg, enrichment_rels)

    print("\nStep 2/5: Exporting feature table ...")
    _export_features(spark, cfg)

    if cfg.test_size is not None:
        print("\nStep 3/5: Skipping external holdout (using internal train/test split) ...")
    else:
        print("\nStep 3/5: Creating holdout split ...")
        _create_holdout(spark, cfg)

    print("\nStep 4/5: Training sklearn classifier ...")
    _train_and_register(cfg)

    if cfg.test_size is None:
        print("\nStep 5/5: Scoring and evaluating ...")
        _score_and_evaluate(spark, cfg)
    else:
        print("\nStep 5/5: Skipping external scoring (test metrics logged internally) ...")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
