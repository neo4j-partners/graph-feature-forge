# Nuclear Option: Bypass AutoML, Train scikit-learn Directly

**Date:** 2026-04-08
**Triggered by:** v0.3.1 failure (run `660291416856599`)
**Full history:** See [TRYAGAIN.md](TRYAGAIN.md) for all 14 runs, bugs found, and catalog migration details.

## What Happened

After 14 consecutive failures at Step 4 (AutoML training), we exhausted every
viable workaround for `databricks.automl.classify()` on runtime 17.3 with
AutoML Service mode enabled:

| Approach | Result |
|----------|--------|
| Pass DataFrame | AutoML Service creates global temp view; internal job can't see it |
| Pass table name with backtick-escaped hyphens | AutoML double-escapes backticks |
| Pass table name without backticks | `INVALID_IDENTIFIER` on hyphenated catalog |
| Rename catalog to underscores + pass DataFrame | Temp view still invisible |
| Rename catalog + persist labeled rows to real table + pass table name string | **Internal validation still rejects data** |

The final attempt (v0.3.1) proved the problem is in AutoML's internal validation,
not our data:

- Table `graph_feature_forge.enrichment.automl_training_input` has **30 rows, 0 nulls,
  10 per class** (Aggressive/Moderate/Conservative)
- AutoML's minimum is 5 rows per class — we have 10
- The cluster's Spark session reads the table correctly
- Yet AutoML's internal job (`supervised_learner.py:_validate_dataset_has_rows`) raises
  `UnsupportedDataError: "Every value in the selected target_col risk_category is either
  null or does not have enough rows (5) per target class"`

**Conclusion:** AutoML Service mode on 17.3+ is fundamentally broken for this use case.
The internal job's preprocessing pipeline misreads valid data through an opaque,
un-debuggable code path.

## Proposed Alternative: Direct scikit-learn Training

Replace `train_automl_classifier()` with `train_sklearn_classifier()` that:

1. Reads the feature table into pandas (102 rows, 130 features)
2. Filters to labeled rows only (30 rows after holdout)
3. Trains 3 models: RandomForest, GradientBoosting, LogisticRegression
4. Selects best by stratified cross-validated F1 (macro)
5. Logs all runs to MLflow (metrics, parameters, model artifact)
6. Returns the best model URI + F1 score

### Why This Works

- **No AutoML Service mode** — no internal jobs, no temp views, no opaque validation
- **No Spark dependency for training** — pandas/scikit-learn only, runs in-process
- **Same MLflow integration** — uses existing `register_model()` / `promote_if_improved()`
- **Same pipeline structure** — drop-in replacement for `train_automl_classifier()`
- **Appropriate for dataset size** — 30 rows is too small for AutoML's model search
  to add value over a simple grid of well-known classifiers

### What Changes

| File | Change |
|------|--------|
| `src/graph_feature_forge/ml/automl_training.py` | Add `train_sklearn_classifier()` function |
| `agent_modules/gds_fastrp_features.py` | Call `train_sklearn_classifier()` instead of `train_automl_classifier()` |
| `agent_modules/gds_community_features.py` | Same swap |
| `agent_modules/gds_baseline_comparison.py` | Same swap (if it uses AutoML) |
| `pyproject.toml` | Version bump to 0.4.0 (new training approach) |

### What Stays the Same

- Steps 1-3 (GDS features, export, holdout) — unchanged, already passing
- Step 5 (scoring, evaluation, writeback) — unchanged
- Model registration and promotion — unchanged (`register_model`, `promote_if_improved`)
- MLflow experiment tracking — unchanged (just manual logging instead of AutoML-generated)
- Pipeline orchestration (`run_pipeline.sh`) — unchanged

### `train_sklearn_classifier` Sketch

```python
def train_sklearn_classifier(
    feature_table: str,
    target_col: str = "risk_category",
    exclude_cols: list[str] | None = None,
    experiment_name: str | None = None,
) -> dict:
    """Train scikit-learn classifiers and return best model info."""
    import mlflow
    from pyspark.sql import SparkSession
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    if exclude_cols is None:
        exclude_cols = ["customer_id"]

    spark = SparkSession.builder.getOrCreate()
    pdf = spark.table(feature_table).toPandas()
    pdf = pdf[pdf[target_col].notna()]

    feature_cols = [c for c in pdf.columns if c not in exclude_cols + [target_col]]
    X = pdf[feature_cols].fillna(0)
    le = LabelEncoder()
    y = le.fit_transform(pdf[target_col])

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }

    best_f1, best_name, best_model = 0, None, None
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=min(5, len(y) // 3), scoring="f1_macro")
        f1 = scores.mean()
        with mlflow.start_run(run_name=name):
            model.fit(X, y)
            mlflow.log_param("classifier", name)
            mlflow.log_metric("val_f1_score", f1)
            mlflow.sklearn.log_model(model, "model")
            if f1 > best_f1:
                best_f1, best_name, best_model = f1, name, model

    print(f"  Best F1: {best_f1:.4f} ({best_name})")
    return {"best_f1": best_f1, "best_model": best_name, "model_uri": ...}
```

## Risks

1. **Fewer model types explored** — only 3 vs AutoML's full search. Acceptable for 30 rows.
2. **No automatic feature preprocessing** — AutoML handles encoding, imputation, scaling.
   We handle this manually (fillna, LabelEncoder). Acceptable for clean numeric data.
3. **Cross-validation on 30 rows** — high variance in F1 estimates. Use stratified k-fold
   with k=3 or k=5 to mitigate.

## Decision

Pending user approval. If approved, implement and bump to v0.4.0.
