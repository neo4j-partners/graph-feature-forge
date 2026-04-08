"""AutoML training utilities for GDS feature engineering notebooks.

Provides reusable functions for holdout simulation, AutoML training,
model registration and promotion, scoring evaluation, and MLflow
experiment comparison.  Used by the ``agent_modules/gds_*`` entry
points.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Holdout simulation
# ---------------------------------------------------------------------------


def create_holdout(
    features_pdf: pd.DataFrame,
    label_col: str = "risk_category",
    holdout_per_class: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified holdout: keep *holdout_per_class* labels per category,
    null the rest.

    Returns ``(holdout_features, ground_truth)`` where *holdout_features*
    has most labels set to ``None`` and *ground_truth* preserves the
    original mapping plus an ``is_held_out`` column so downstream phases
    can reproduce the exact same split.
    """
    np.random.seed(seed)

    labeled_mask = features_pdf[label_col].notna() & (features_pdf[label_col] != "")
    labeled_df = features_pdf[labeled_mask]

    keep_indices: list[int] = []
    for category in labeled_df[label_col].unique():
        cat_indices = labeled_df[labeled_df[label_col] == category].index.tolist()
        keep = np.random.choice(
            cat_indices,
            size=min(holdout_per_class, len(cat_indices)),
            replace=False,
        )
        keep_indices.extend(keep)

    holdout_mask = labeled_mask & ~features_pdf.index.isin(keep_indices)

    ground_truth = features_pdf[["customer_id", label_col]].copy()
    ground_truth.columns = ["customer_id", f"true_{label_col}"]
    ground_truth["is_held_out"] = holdout_mask.values

    features_pdf = features_pdf.copy()
    features_pdf.loc[holdout_mask, label_col] = None

    kept = features_pdf[label_col].notna().sum()
    held_out = holdout_mask.sum()
    print(f"  Holdout: {kept} labels kept, {held_out} held out")

    return features_pdf, ground_truth


def reapply_holdout(
    features_pdf: pd.DataFrame,
    ground_truth_pdf: pd.DataFrame,
    label_col: str = "risk_category",
) -> pd.DataFrame:
    """Re-apply a previous holdout split using the ``is_held_out`` column
    from the ground truth table saved by :func:`create_holdout`."""
    if "is_held_out" not in ground_truth_pdf.columns:
        raise ValueError(
            "Column 'is_held_out' not found in ground truth table. "
            f"Available columns: {ground_truth_pdf.columns.tolist()}. "
            "Run gds_fastrp_features.py first to create the holdout split."
        )

    if not pd.api.types.is_bool_dtype(ground_truth_pdf["is_held_out"]):
        raise TypeError(
            f"Column 'is_held_out' has unexpected dtype "
            f"'{ground_truth_pdf['is_held_out'].dtype}'; expected bool. "
            "The ground truth table may be corrupted."
        )

    num_held_out = ground_truth_pdf["is_held_out"].sum()
    if num_held_out == 0:
        raise ValueError(
            "No records marked as held out in the ground truth table. "
            "Re-run gds_fastrp_features.py to regenerate the holdout split."
        )

    held_out_ids = set(
        ground_truth_pdf[ground_truth_pdf["is_held_out"]]["customer_id"]
    )
    features_pdf = features_pdf.copy()
    features_pdf.loc[
        features_pdf["customer_id"].isin(held_out_ids), label_col
    ] = None
    kept = features_pdf[label_col].notna().sum()
    held_out = len(held_out_ids)
    print(f"  Re-applied holdout: {kept} labels kept, {held_out} held out")
    return features_pdf


# ---------------------------------------------------------------------------
# AutoML training
# ---------------------------------------------------------------------------


def train_automl_classifier(
    feature_table: str,
    target_col: str = "risk_category",
    exclude_cols: list[str] | None = None,
    experiment_name: str | None = None,
    timeout_minutes: int = 30,
) -> Any:
    """Run ``databricks.automl.classify`` and return the summary."""
    from databricks import automl

    if exclude_cols is None:
        exclude_cols = ["customer_id"]

    # AutoML stores experiments under /Users/<username>/databricks_automl/<experiment_name>.
    # Ensure the parent directory exists so the experiment can be created.
    if experiment_name:
        from databricks.sdk import WorkspaceClient

        ws = WorkspaceClient()
        username = ws.current_user.me().user_name
        full_path = f"/Users/{username}/databricks_automl/{experiment_name}"
        parent_dir = full_path.rsplit("/", 1)[0]
        ws.workspace.mkdirs(parent_dir)

    # AutoML Service mode can't see global temp views created from DataFrames.
    # Persist labeled-only rows to a real table and pass the table name as a string.
    from pyspark.sql import SparkSession, functions as F

    spark = SparkSession.builder.getOrCreate()
    labeled_df = spark.table(feature_table).filter(F.col(target_col).isNotNull())

    training_table = feature_table.replace("customer_graph_features", "automl_training_input")
    labeled_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(training_table)
    row_count = spark.table(training_table).count()
    print(f"  Training table: {training_table} ({row_count} labeled rows)")

    summary = automl.classify(
        dataset=training_table,
        target_col=target_col,
        primary_metric="f1",
        exclude_cols=exclude_cols,
        timeout_minutes=timeout_minutes,
        experiment_name=experiment_name,
    )

    print(f"  Best F1: {summary.best_trial.evaluation_metric_score:.4f}")
    print(f"  Best model: {summary.best_trial.model_description}")
    print(f"  Trials: {len(summary.trials)}")
    return summary


# ---------------------------------------------------------------------------
# Model registration and promotion
# ---------------------------------------------------------------------------


def register_model(
    model_uri: str,
    model_name: str,
    set_champion: bool = True,
) -> Any:
    """Register a model in Unity Catalog and optionally set Champion alias."""
    import mlflow

    mlflow.set_registry_uri("databricks-uc")
    registered = mlflow.register_model(model_uri, model_name)
    print(f"  Registered: {model_name} v{registered.version}")

    if set_champion:
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(model_name, "Champion", registered.version)
        print(f"  Set Champion alias -> v{registered.version}")

    return registered


def promote_if_improved(
    model_uri: str,
    model_name: str,
    new_f1: float,
) -> bool:
    """Register a new model version and promote to Champion only if F1
    improves over the current Champion.

    Returns ``True`` if the new version was promoted.
    """
    import mlflow

    mlflow.set_registry_uri("databricks-uc")
    registered = mlflow.register_model(model_uri, model_name)
    client = mlflow.MlflowClient()

    try:
        current_champion = client.get_model_version_by_alias(model_name, "Champion")
        champion_run = client.get_run(current_champion.run_id)
        champion_f1 = champion_run.data.metrics.get("val_f1_score", 0)
    except Exception:
        champion_f1 = 0

    print(f"  Current Champion F1: {champion_f1:.4f}")
    print(f"  New model F1:        {new_f1:.4f}")

    if new_f1 > champion_f1:
        client.set_registered_model_alias(model_name, "Champion", registered.version)
        print(f"  Promoted v{registered.version} to Champion")
        return True

    print("  Current Champion retained")
    return False


# ---------------------------------------------------------------------------
# Scoring evaluation
# ---------------------------------------------------------------------------


def evaluate_predictions(
    spark: Any,
    predictions_df: Any,
    ground_truth_table: str,
) -> float:
    """Join predictions against ground truth and return accuracy."""
    from pyspark.sql import functions as F

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

    print(f"  Accuracy on held-out: {correct}/{total} = {accuracy:.2%}")
    return accuracy


# ---------------------------------------------------------------------------
# kNN visualization
# ---------------------------------------------------------------------------


def run_knn_analysis(
    gds: Any,
    G: Any,
    spotlight_customers: list[str] | None = None,
) -> None:
    """Run kNN on FastRP embeddings and print nearest-neighbor analysis."""
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
    print(f"  kNN relationships created: {knn_result['relationshipsWritten']}")

    if spotlight_customers:
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
                print(f"\n  {name} (community {neighbors.iloc[0]['community']}):")
                print(neighbors.to_string(index=False))

    # Community overlap summary
    overlap = gds.run_cypher("""
        MATCH (c:Customer)-[r:SIMILAR_TO]->(n:Customer)
        RETURN
            CASE WHEN c.community_id = n.community_id THEN 'same' ELSE 'different' END AS community_match,
            count(*) AS count,
            avg(r.similarity) AS avg_similarity
    """)
    print("\n  kNN community overlap:")
    print(overlap.to_string(index=False))


# ---------------------------------------------------------------------------
# MLflow experiment comparison
# ---------------------------------------------------------------------------


def compare_experiments(
    experiments: dict[str, str],
) -> pd.DataFrame:
    """Print a comparison table of best F1 across MLflow experiments.

    *experiments* maps display labels to experiment paths.
    Returns a DataFrame with the results.
    """
    import mlflow

    rows = []
    print("=" * 70)
    print(f"{'Experiment':<30s} {'Best F1':>10s} {'Best Model':<25s}")
    print("=" * 70)

    for label, path in experiments.items():
        try:
            exp = mlflow.get_experiment_by_name(path)
            if exp is None:
                print(f"{label:<30s} {'N/A':>10s} {'(not yet run)':<25s}")
                rows.append({"experiment": label, "f1": None, "model": None})
                continue

            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["metrics.val_f1_score DESC"],
                max_results=1,
            )
            if runs.empty:
                print(f"{label:<30s} {'N/A':>10s} {'(no runs)':<25s}")
                rows.append({"experiment": label, "f1": None, "model": None})
                continue

            best_f1 = runs.iloc[0]["metrics.val_f1_score"]
            best_model = runs.iloc[0].get("params.classifier", "unknown")
            print(f"{label:<30s} {best_f1:>10.4f} {best_model:<25s}")
            rows.append({"experiment": label, "f1": best_f1, "model": best_model})
        except Exception as e:
            print(f"{label:<30s} {'error':>10s} {str(e)[:25]:<25s}")
            rows.append({"experiment": label, "f1": None, "model": str(e)[:50]})

    print("=" * 70)
    return pd.DataFrame(rows)


def extract_feature_importance(
    experiment_path: str,
    top_n: int = 20,
) -> pd.Series | None:
    """Load the best model from an MLflow experiment and return feature
    importances, or ``None`` if unavailable.
    """
    import mlflow

    exp = mlflow.get_experiment_by_name(experiment_path)
    if exp is None:
        return None

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.val_f1_score DESC"],
        max_results=1,
    )
    if runs.empty:
        return None

    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model"

    try:
        model = mlflow.sklearn.load_model(model_uri)

        if hasattr(model, "named_steps"):
            estimator = list(model.named_steps.values())[-1]
        else:
            estimator = model

        if not hasattr(estimator, "feature_importances_"):
            print("  Model does not expose feature_importances_")
            return None

        feature_names = (
            estimator.feature_names_in_
            if hasattr(estimator, "feature_names_in_")
            else range(len(estimator.feature_importances_))
        )
        importances = pd.Series(
            estimator.feature_importances_, index=feature_names
        ).sort_values(ascending=False)

        print(f"  Top {top_n} features:")
        print(importances.head(top_n).to_string())

        graph_in_top = sum(
            1
            for f in importances.head(top_n).index
            if str(f).startswith("fastrp_") or f == "community_id"
        )
        print(f"\n  Graph features in top {top_n}: {graph_in_top}/{top_n}")
        return importances

    except Exception as e:
        print(f"  Could not load model: {e}")
        return None
