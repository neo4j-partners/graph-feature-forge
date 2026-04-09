# Databricks notebook source
# MAGIC %md
# MAGIC # GDS Baseline Comparison
# MAGIC
# MAGIC Trains a model using only tabular customer attributes (no graph features) to
# MAGIC establish a baseline. Compares against the FastRP-only and FastRP + Louvain
# MAGIC results in MLflow.
# MAGIC
# MAGIC **Prerequisite:** Run `gds_fastrp_features` first. This notebook reuses the
# MAGIC same holdout split and feature table.
# MAGIC
# MAGIC **Runtime requirement:** Databricks Runtime 17.x LTS ML or earlier.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os

CATALOG = os.environ.get("CATALOG_NAME", "graph_feature_forge")
SCHEMA = os.environ.get("SCHEMA_NAME", "enrichment")
FEATURE_TABLE = f"`{CATALOG}`.`{SCHEMA}`.customer_graph_features"

EMBEDDING_DIM = 128

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train a Tabular-Only Model
# MAGIC
# MAGIC Exclude all graph-derived columns (128 FastRP dimensions, community_id) and
# MAGIC train on tabular features only: annual_income, credit_score.

# COMMAND ----------

from databricks import automl

# Columns to exclude: customer_id + all graph-derived features
graph_feature_cols = [f"fastrp_{i}" for i in range(EMBEDDING_DIM)] + ["community_id"]
exclude_cols = ["customer_id"] + graph_feature_cols

summary = automl.classify(
    dataset=FEATURE_TABLE,
    target_col="risk_category",
    primary_metric="f1",
    exclude_cols=exclude_cols,
    timeout_minutes=30,
    experiment_name="/Shared/graph_feature_forge/tabular_only_baseline",
)

# COMMAND ----------

print(f"Tabular-only best F1: {summary.best_trial.evaluation_metric_score:.4f}")
print(f"Best model: {summary.best_trial.model_description}")
print(f"Number of trials: {len(summary.trials)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare All Three Experiments in MLflow

# COMMAND ----------

import mlflow

experiments = {
    "FastRP only": "/Shared/graph_feature_forge/fastrp_risk_classification",
    "FastRP + Louvain": "/Shared/graph_feature_forge/fastrp_louvain_risk_classification",
    "Tabular only": "/Shared/graph_feature_forge/tabular_only_baseline",
}

print("=" * 70)
print(f"{'Experiment':<30s} {'Best F1':>10s} {'Best Model':<25s}")
print("=" * 70)

for label, path in experiments.items():
    try:
        exp = mlflow.get_experiment_by_name(path)
        if exp is None:
            print(f"{label:<30s} {'N/A':>10s} {'(not yet run)':<25s}")
            continue

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.val_f1_score DESC"],
            max_results=1,
        )

        if runs.empty:
            print(f"{label:<30s} {'N/A':>10s} {'(no runs)':<25s}")
            continue

        best_f1 = runs.iloc[0]["metrics.val_f1_score"]
        best_model = runs.iloc[0].get("params.classifier", "unknown")
        print(f"{label:<30s} {best_f1:>10.4f} {best_model:<25s}")
    except Exception as e:
        print(f"{label:<30s} {'error':>10s} {str(e)[:25]:<25s}")

print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interpretation
# MAGIC
# MAGIC The F1 delta between tabular-only and graph-augmented is the core result.
# MAGIC If graph features carry signal:
# MAGIC
# MAGIC - FastRP should outperform tabular-only
# MAGIC - FastRP + Louvain may further improve over FastRP alone
# MAGIC
# MAGIC If the delta is small or negative with 103 customers, that is a valid result.
# MAGIC Small datasets may not provide enough signal for graph features to differentiate.
# MAGIC The SEC EDGAR validation in Phase 5 addresses this at scale.
# MAGIC
# MAGIC Open the MLflow experiment comparison UI to see metric differences, parameter
# MAGIC differences, and feature importance plots side by side. Select the best run
# MAGIC from each experiment and click "Compare."

# COMMAND ----------

# Feature importance comparison: which features matter in each model?
# Load the FastRP + Louvain best run to check feature importance
graph_exp = mlflow.get_experiment_by_name(
    "/Shared/graph_feature_forge/fastrp_louvain_risk_classification"
)

if graph_exp:
    graph_runs = mlflow.search_runs(
        experiment_ids=[graph_exp.experiment_id],
        order_by=["metrics.val_f1_score DESC"],
        max_results=1,
    )

    if not graph_runs.empty:
        run_id = graph_runs.iloc[0]["run_id"]
        model_uri = f"runs:/{run_id}/model"

        try:
            model = mlflow.sklearn.load_model(model_uri)

            # Extract the final estimator from the pipeline
            if hasattr(model, "named_steps"):
                estimator = list(model.named_steps.values())[-1]
            else:
                estimator = model

            if hasattr(estimator, "feature_importances_"):
                import pandas as pd

                feature_names = (
                    estimator.feature_names_in_
                    if hasattr(estimator, "feature_names_in_")
                    else range(len(estimator.feature_importances_))
                )
                importances = pd.Series(
                    estimator.feature_importances_, index=feature_names
                ).sort_values(ascending=False)

                print("Top 20 most important features (graph-augmented best model):")
                print(importances.head(20).to_string())

                # Count how many graph features are in the top 20
                graph_in_top20 = sum(
                    1
                    for f in importances.head(20).index
                    if f.startswith("fastrp_") or f == "community_id"
                )
                print(f"\nGraph features in top 20: {graph_in_top20}/20")
            else:
                print("Best model does not expose feature_importances_ (e.g. logistic regression)")
                print("Enable SHAP in the generated notebook for model-agnostic importance.")
        except Exception as e:
            print(f"Could not load model for feature importance: {e}")
