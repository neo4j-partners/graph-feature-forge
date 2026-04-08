"""Tabular-only baseline: train on annual_income + credit_score, compare all three experiments.

Trains a model using only tabular customer attributes (no graph features) to
establish a baseline. Produces a three-way MLflow comparison across FastRP-only,
FastRP + Louvain, and tabular-only experiments, plus feature importance analysis.

Prerequisite: run ``gds_fastrp_features.py`` first — this reuses the same
holdout split and feature table.

Usage:
    python -m cli upload --wheel
    python -m cli upload gds_baseline_comparison.py
    python -m cli submit gds_baseline_comparison.py --compute cluster
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BaselineConfig:
    """Configuration from environment variables and CLI flags."""

    catalog: str
    schema: str
    embedding_dim: int

    @property
    def feature_table(self) -> str:
        return f"`{self.catalog}`.`{self.schema}`.customer_graph_features"

    @classmethod
    def from_env(cls) -> BaselineConfig:
        from graph_feature_forge import inject_params

        inject_params()

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--embedding-dim", type=int, default=128)
        flags, _ = parser.parse_known_args()

        return cls(
            catalog=os.getenv("CATALOG_NAME", "graph_feature_forge"),
            schema=os.getenv("SCHEMA_NAME", "enrichment"),
            embedding_dim=flags.embedding_dim,
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


def _train_tabular_baseline(cfg: BaselineConfig) -> None:
    """Train AutoML on tabular features only (exclude all graph columns)."""
    from graph_feature_forge.ml.automl_training import train_automl_classifier

    graph_feature_cols = [f"fastrp_{i}" for i in range(cfg.embedding_dim)] + ["community_id"]
    exclude_cols = ["customer_id"] + graph_feature_cols

    train_automl_classifier(
        feature_table=cfg.feature_table,
        exclude_cols=exclude_cols,
        experiment_name="/Shared/graph-feature-forge/tabular_only_baseline",
    )


def _compare_all() -> None:
    """Three-way MLflow experiment comparison."""
    from graph_feature_forge.ml.automl_training import compare_experiments

    compare_experiments({
        "FastRP only": "/Shared/graph-feature-forge/fastrp_risk_classification",
        "FastRP + Louvain": "/Shared/graph-feature-forge/fastrp_louvain_risk_classification",
        "Tabular only": "/Shared/graph-feature-forge/tabular_only_baseline",
    })


def _feature_importance() -> None:
    """Extract and print feature importance from the graph-augmented model."""
    from graph_feature_forge.ml.automl_training import extract_feature_importance

    extract_feature_importance(
        experiment_path="/Shared/graph-feature-forge/fastrp_louvain_risk_classification",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = BaselineConfig.from_env()

    print("=" * 60)
    print("GDS Baseline Comparison")
    print("=" * 60)

    _authenticate()

    print("\nStep 1/3: Training tabular-only baseline ...")
    _train_tabular_baseline(cfg)

    print("\nStep 2/3: Three-way experiment comparison ...")
    _compare_all()

    print("\nStep 3/3: Feature importance analysis ...")
    _feature_importance()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
