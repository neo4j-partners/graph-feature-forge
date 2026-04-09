"""GDS Community Detection: FastRP + Louvain → export → retrain → promote → kNN.

Adds Louvain community detection as a categorical feature on top of FastRP
embeddings. Retrains AutoML with the combined feature set and promotes the
Champion model only if F1 improves.

Prerequisite: run ``gds_fastrp_features.py`` first — this reuses the holdout
split and ground truth table created there.

Usage:
    python -m cli upload --wheel
    python -m cli upload gds_community_features.py
    python -m cli submit gds_community_features.py --compute cluster
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
class GDSCommunityConfig:
    """Configuration from environment variables and CLI flags."""

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    catalog: str
    schema: str
    embedding_dim: int
    test_size: float | None
    pca_components: int | None

    @property
    def feature_table(self) -> str:
        return f"{self.catalog}.{self.schema}.customer_graph_features"

    @property
    def enrichment_log_table(self) -> str:
        return f"{self.catalog}.{self.schema}.enrichment_log"

    @property
    def model_name(self) -> str:
        return f"{self.catalog}.{self.schema}.graph_feature_forge_risk_classifier"

    @classmethod
    def from_env(cls) -> GDSCommunityConfig:
        from graph_feature_forge import inject_params

        inject_params()

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--embedding-dim", type=int, default=128)
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
            test_size=test_size,
            pca_components=pca_components,
        )


# ---------------------------------------------------------------------------
# Authentication
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
    except Exception:
        print("  No enrichment log found")
        return []


def _compute_features(cfg: GDSCommunityConfig, enrichment_rel_types: list[str]) -> Any:
    """Project graph, compute FastRP + Louvain, write to Neo4j.

    Returns the GDS graph object for kNN analysis.
    """
    from graphdatascience import GraphDataScience

    from graph_feature_forge.ml.feature_engineering import (
        PROJECTION_NAME,
        BASE_RELATIONSHIP_SPEC,
        NODE_LABELS,
    )

    gds = GraphDataScience(
        cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password),
        database=cfg.neo4j_database,
    )
    print(f"  GDS version: {gds.version()}")

    relationship_spec = dict(BASE_RELATIONSHIP_SPEC)
    for rel_type in enrichment_rel_types:
        relationship_spec[rel_type] = {"orientation": "UNDIRECTED"}

    try:
        gds.graph.drop(gds.graph.get(PROJECTION_NAME))
    except Exception:
        pass

    G, _ = gds.graph.project(PROJECTION_NAME, NODE_LABELS, relationship_spec)
    print(f"  Projected — Nodes: {G.node_count()}, Relationships: {G.relationship_count()}")

    fastrp_result = gds.fastRP.mutate(
        G,
        mutateProperty="fastrp_embedding",
        embeddingDimension=cfg.embedding_dim,
        iterationWeights=[0.0, 1.0, 1.0, 0.8, 0.5],
        randomSeed=42,
    )
    print(f"  FastRP: {fastrp_result['nodePropertiesWritten']} nodes")

    louvain_result = gds.louvain.mutate(
        G,
        mutateProperty="community_id",
        maxLevels=10,
        maxIterations=10,
    )
    print(f"  Louvain: {louvain_result['communityCount']} communities, modularity {louvain_result['modularity']:.4f}")

    gds.graph.nodeProperties.write(G, node_properties=["fastrp_embedding"], node_labels=["Customer"])
    gds.graph.nodeProperties.write(G, node_properties=["community_id"], node_labels=["Customer"])
    print("  Properties written to Neo4j")

    return gds, G


def _export_features_with_community(spark: Any, cfg: GDSCommunityConfig) -> None:
    """Export FastRP + community_id to the Delta feature table."""
    from graph_feature_forge.ml.feature_engineering import export_feature_table

    export_feature_table(
        spark=spark,
        uri=cfg.neo4j_uri,
        username=cfg.neo4j_username,
        password=cfg.neo4j_password,
        database=cfg.neo4j_database,
        catalog=cfg.catalog,
        schema=cfg.schema,
    )


def _reapply_holdout(spark: Any, cfg: GDSCommunityConfig) -> None:
    """Re-apply the holdout split from gds_fastrp_features using the
    ``is_held_out`` column saved in the ground truth table."""
    from graph_feature_forge.ml.automl_training import reapply_holdout

    ground_truth_table = f"`{cfg.catalog}`.`{cfg.schema}`.holdout_ground_truth"
    ground_truth_pdf = spark.table(ground_truth_table).toPandas()
    features_pdf = spark.table(cfg.feature_table).toPandas()

    features_pdf = reapply_holdout(features_pdf, ground_truth_pdf)
    spark.createDataFrame(features_pdf).write.mode("overwrite").saveAsTable(cfg.feature_table)


def _train_and_promote(cfg: GDSCommunityConfig, summary: Any) -> None:
    """Promote to Champion only if F1 improves."""
    from graph_feature_forge.ml.automl_training import promote_if_improved

    promote_if_improved(
        model_uri=summary.best_trial.model_path,
        model_name=cfg.model_name,
        new_f1=summary.best_trial.evaluation_metric_score,
    )


def _run_knn(gds: Any, G: Any) -> None:
    """Run kNN nearest-neighbor analysis on the projection."""
    from graph_feature_forge.ml.automl_training import run_knn_analysis

    run_knn_analysis(
        gds=gds,
        G=G,
        spotlight_customers=["James Anderson", "Maria Rodriguez", "Robert Chen"],
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = GDSCommunityConfig.from_env()

    print("=" * 60)
    print("GDS Community Detection Features")
    print("=" * 60)

    _authenticate()

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    print("\nStep 1/5: Computing FastRP + Louvain ...")
    enrichment_rels = _discover_enrichment_rels(spark, cfg.enrichment_log_table)
    gds, G = _compute_features(cfg, enrichment_rels)

    print("\nStep 2/5: Exporting feature table (with community_id) ...")
    _export_features_with_community(spark, cfg)

    if cfg.test_size is not None:
        print("\nStep 3/5: Skipping external holdout (using internal train/test split) ...")
    else:
        print("\nStep 3/5: Re-applying holdout split ...")
        _reapply_holdout(spark, cfg)

    print("\nStep 4/5: Retraining with combined features ...")
    from graph_feature_forge.ml.automl_training import train_sklearn_classifier

    summary = train_sklearn_classifier(
        feature_table=cfg.feature_table,
        experiment_name="/Shared/graph_feature_forge/fastrp_louvain_risk_classification",
        test_size=cfg.test_size,
        pca_components=cfg.pca_components,
    )
    _train_and_promote(cfg, summary)

    print("\nStep 5/5: kNN nearest-neighbor analysis ...")
    _run_knn(gds, G)

    G.drop()
    print("\nGDS projection dropped")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
