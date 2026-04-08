# Plan: Replace AutoML with scikit-learn Training

**Date:** 2026-04-08
**Status:** Phase 2.5 complete — graph features add value with PCA, cv_F1=0.801
**Replaces:** `databricks.automl.classify()` (broken on DBR 17.3 Service mode)
**Background:** [NUCLEAR.md](../NUCLEAR.md)

---

## What We're Doing

We're replacing the AutoML training step (Step 4 of the pipeline) with a
straightforward scikit-learn training function. Everything before Step 4
(graph features, export, holdout) and everything after Step 4 (scoring,
evaluation, writeback) stays the same. We're only swapping out the part
that trains the model.

The new function will:

- Read the feature table from Delta into pandas
- Filter to labeled rows only
- Train three classifiers (RandomForest, GradientBoosting, LogisticRegression)
- Pick the best one by stratified cross-validated F1 score
- Log everything to MLflow automatically using autolog
- Return the best model's URI and F1 score so the rest of the pipeline can
  register and promote it as before

---

## Why This Approach

- **30 rows is too small for AutoML's model search to add value.** A simple
  grid of well-known classifiers is the right tool for this dataset size.
- **Matches Databricks reference patterns exactly.** The official sklearn docs,
  the ai-dev-kit skills, and the Hyperopt model-selection notebook all use this
  same train-compare-log loop.
- **Uses `mlflow.sklearn.autolog()`.** This is Databricks' recommended approach.
  One line replaces all manual param/metric/model logging and adds input
  examples and model signatures automatically.
- **No new infrastructure.** Runs in-process on the existing cluster. No
  AutoML Service mode, no internal jobs, no temp views, no opaque validation.

---

## Phased Approach

We're doing this in three phases. Each phase is independently testable and
deployable. If Phase 1 works, the pipeline is unblocked. Phases 2 and 3
are improvements we can make later.

### Phase 1: Basic Implementation (unblock the pipeline)

Get a working `train_sklearn_classifier()` function that produces a registered
model. This is the minimum viable replacement.

**What to build:**

- A new function `train_sklearn_classifier()` in `automl_training.py` alongside
  the existing `train_automl_classifier()` (don't delete the old one yet)
- The function reads the feature table, filters to labeled rows, trains three
  classifiers with default hyperparameters, picks the best by cross-validated
  F1, and returns the model URI and score
- Uses `mlflow.sklearn.autolog()` so all params, metrics, and the model artifact
  are logged automatically
- Wraps each classifier in a `Pipeline` with `StandardScaler` so features are
  normalized before training (FastRP embeddings and income/credit score have
  very different scales)
- Uses `StratifiedKFold` with k=3 (not k=5) because we only have about 10
  samples per class — k=3 gives us folds of roughly 7 train / 3 test per class
  which is the minimum reasonable split

**What to change in the callers:**

- `gds_fastrp_features.py` — swap `train_automl_classifier()` call to
  `train_sklearn_classifier()`
- `gds_community_features.py` — same swap
- `gds_baseline_comparison.py` — same swap if it uses AutoML

**What to verify:**

- The function runs without errors on the cluster
- MLflow experiment shows logged runs with params, metrics, and a model artifact
- The returned model URI can be passed to `register_model()` and
  `promote_if_improved()` without changes
- The registered model can be loaded by `score_unlabeled_customers()` for
  inference

**Version bump:** 0.4.0 in pyproject.toml (new training approach)

### Phase 2: Validation and Confidence (make sure it's good)

Once Phase 1 is deployed and the pipeline runs end-to-end, we come back and
check whether the model is actually useful.

**What to do:**

- Run the full pipeline and check the MLflow experiment for all three
  classifiers' F1 scores — are they reasonable? (Above 0.5 means better than
  random for 3 classes)
- Check the held-out accuracy from `evaluate_predictions()` — does the model
  generalize at all to the held-out labels?
- Use `extract_feature_importance()` (already in the codebase) to check whether
  graph features (FastRP dimensions, community_id) are actually contributing
  to predictions or whether the model is just using income and credit score
- Compare the three experiment runs (FastRP, Community, Baseline) using
  `compare_experiments()` to see if graph features help
- If F1 is very low or feature importances look wrong, investigate whether
  the feature table has data quality issues (all-zero FastRP columns, single
  community, etc.)

**No code changes needed.** This is analysis work using functions that already
exist in the codebase.

### Phase 3: Iteration and Hardening (make it better)

Based on what we learn in Phase 2, make targeted improvements.

**Potential improvements (pick based on Phase 2 findings):**

- Add hyperparameter tuning with Optuna if the default hyperparameters
  underperform — Databricks recommends Optuna on DBR 17.x and MLflow 3.0
  has native integration via `MlflowStorage`
- Add more classifiers (XGBoost, LightGBM) if the dataset grows beyond 30 rows
- Add a `Pipeline` step for one-hot encoding `community_id` instead of treating
  it as a numeric feature
- Add class weights or SMOTE if the class distribution turns out to be
  imbalanced (right now it's 10 per class, but that could change)
- Wire up the Optuna reference notebook pattern from Databricks docs if we
  want distributed tuning later

---

## Files That Change

| File | Phase | What Changes |
|------|-------|-------------|
| `src/graph_feature_forge/ml/automl_training.py` | 1 | Add `train_sklearn_classifier()` function |
| `src/graph_feature_forge/ml/automl_training.py` | 2.5 | Add `test_size`, `pca_components` params; ColumnTransformer; per-class metrics |
| `agent_modules/gds_fastrp_features.py` | 1 | Call `train_sklearn_classifier()` instead of `train_automl_classifier()` |
| `agent_modules/gds_fastrp_features.py` | 2.5 | Add `TEST_SIZE`/`PCA_COMPONENTS` config; conditionally skip `create_holdout()` |
| `agent_modules/gds_community_features.py` | 1, 2.5 | Same as fastrp |
| `agent_modules/gds_baseline_comparison.py` | 1, 2.5 | Same as fastrp |
| `pyproject.toml` | 1→2.5 | Version bump 0.4.0 → 0.5.0 |
| `.env.example` | 2.5 | Add `TEST_SIZE=0.2` and `PCA_COMPONENTS=5` |
| `run_pipeline.sh` | — | No change expected (orchestration is the same) |

## Files That Don't Change

- `feature_engineering.py` — Steps 1-3 are untouched
- `register_model()` / `promote_if_improved()` — model registration stays the same
- `score_unlabeled_customers()` — scoring stays the same (already uses `mlflow.pyfunc`)
- `evaluate_predictions()` — updated in 2.5 to optionally log to MLflow, but signature is backward-compatible
- `compare_experiments()` / `extract_feature_importance()` — already work with sklearn models

---

## Todo List

### Phase 1: Basic Implementation

- [x] Write `train_sklearn_classifier()` in `automl_training.py`
  - [x] Read feature table into pandas, filter to labeled rows
  - [x] Build feature matrix (exclude customer_id and target column)
  - [x] Encode target labels with LabelEncoder
  - [x] Create three pipelines (StandardScaler + each classifier)
  - [x] Run StratifiedKFold cross-validation (k=3) for each pipeline
  - [x] Pick best by mean F1 macro score
  - [x] Fit best model on full labeled set
  - [x] Use mlflow.sklearn.autolog() for logging
  - [x] Return summary object with best_trial.model_path and best_trial.evaluation_metric_score (compatible with existing callers)
- [x] Update `gds_fastrp_features.py` to call new function
- [x] Update `gds_community_features.py` to call new function
- [x] Update `gds_baseline_comparison.py` to call new function
- [x] Bump version to 0.4.0 in `pyproject.toml` (then 0.4.1 to bust wheel cache)
- [x] Build wheel and upload to Databricks volume
- [x] Run pipeline end-to-end on cluster

### Phase 1: Status Checks

After the pipeline run, verify each of these:

- [x] Pipeline completes Step 4 without errors (check job run logs)
- [x] MLflow experiment exists and has 3 child runs (one per classifier)
- [x] Each run has logged params (classifier name, n_estimators, etc.)
- [x] Each run has logged metrics (training_score, F1, accuracy, etc.)
- [x] Each run has a logged model artifact
- [x] Best model URI is valid (can be passed to register_model)
- [x] Model registers in Unity Catalog without errors
- [x] Step 5 (scoring) loads the model and scores unlabeled customers
- [x] Predictions are written back to Neo4j

### Phase 1: Results (run 203387533504795)

| Classifier | F1 (macro) | CV Std |
|------------|-----------|--------|
| **GradientBoosting** | **0.6358** | 0.1856 |
| RandomForest | 0.4902 | 0.0817 |
| LogisticRegression | 0.3293 | 0.0556 |

- **Held-out accuracy: 53/72 = 73.61%** (random baseline ~33%)
- Model registered as Champion v3 in Unity Catalog
- 72 predictions written back to Neo4j

#### Bug fixed during Phase 1

The initial implementation used `LabelEncoder` to convert string labels
(Aggressive/Conservative/Moderate) to integers (0/1/2) for training. This
caused the model to predict `"0"/"1"/"2"` at scoring time instead of the
original string labels, producing 0% held-out accuracy. Fix: removed
`LabelEncoder` and passed string labels directly to sklearn classifiers,
which encode internally and return original strings from `predict()`.

Also discovered that Databricks caches wheels by filename — bumping the
version from 0.4.0 to 0.4.1 was required to force the cluster to pick up
the fixed wheel.

### Phase 2: Validation

- [x] Review F1 scores across all three classifiers — are they above 0.5?
- [x] Check held-out accuracy from evaluate_predictions()
- [x] Run extract_feature_importance() — are graph features in top 20?
- [x] Run compare_experiments() across FastRP / Community / Baseline
- [x] Document findings and decide if Phase 3 improvements are needed

### Phase 2: Results

#### Three-way experiment comparison

| Experiment | Best F1 | Best Model | Features |
|------------|---------|------------|----------|
| **Tabular only** | **0.8527** | LogisticRegression | 2 features (annual_income, credit_score) |
| FastRP only | 0.6358 | GradientBoosting | 130 features (2 tabular + 128 FastRP dims) |
| FastRP + Louvain | 0.6358 | GradientBoosting | 131 features (2 tabular + 128 FastRP + community_id) |

#### FastRP experiment detail (held-out accuracy: 73.6%)

| Classifier | F1 (macro) | CV Std |
|------------|-----------|--------|
| GradientBoosting | 0.6358 | 0.1856 |
| RandomForest | 0.4902 | 0.0817 |
| LogisticRegression | 0.3293 | 0.0556 |

#### Community experiment detail (Champion retained — same F1)

| Classifier | F1 (macro) | CV Std |
|------------|-----------|--------|
| GradientBoosting | 0.6358 | 0.1856 |
| RandomForest | 0.5086 | 0.2020 |
| LogisticRegression | 0.3240 | 0.0623 |

#### Tabular-only baseline detail

| Classifier | F1 (macro) | CV Std |
|------------|-----------|--------|
| LogisticRegression | 0.8527 | 0.0736 |
| RandomForest | 0.8347 | 0.0435 |
| GradientBoosting | 0.7400 | 0.0548 |

#### Feature importance (FastRP experiment, top 20)

Graph features in top 20: **0 out of 20**. The model relies entirely on
features at indices 0 and 1 (annual_income and credit_score), which together
account for ~69% of importance. FastRP dimensions and community_id contribute
negligibly.

#### Key Findings

1. **Graph features hurt performance.** Tabular-only (F1=0.853) beats
   FastRP (F1=0.636) by a wide margin. Adding 128 FastRP embedding dimensions
   and community_id to 30 training rows introduces noise that overwhelms
   the two strong tabular signals.

2. **This is a classic curse-of-dimensionality problem.** 130 features on
   30 rows means the models are fitting noise. The tabular baseline has 2
   features on 30 rows, which is a much healthier ratio.

3. **Community features add nothing.** FastRP-only and FastRP+Louvain have
   identical best F1 (0.6358). The community_id feature has no predictive
   value for risk classification.

4. **LogisticRegression wins when features are clean.** On tabular-only data,
   the simplest model (LogisticRegression) beats both tree ensembles. On
   high-dimensional noisy data (FastRP), GradientBoosting wins because it
   can ignore irrelevant features via tree splits.

5. **The sklearn pipeline works correctly.** All three experiments trained,
   logged, compared, and registered without issues. The infrastructure is
   solid.

#### Recommendations for Phase 3

The graph features aren't helping because there are too many dimensions
for the dataset size. Before adding complexity (Optuna, more classifiers),
consider:

- **Dimensionality reduction on FastRP**: Use PCA to reduce 128 FastRP dims
  to 3-5 components before training. This preserves graph signal while
  reducing noise.
- **Feature selection**: Use SelectKBest or mutual information to keep only
  the FastRP dimensions that correlate with risk_category.
- **More labeled data**: 30 rows is the root constraint. If more labels
  become available, re-evaluate — graph features may become useful at 100+
  labeled rows.
- **Skip graph features for now**: The tabular-only model (F1=0.853) is the
  best performing. Consider using it as Champion until the dataset grows.

### Phase 2.5: Proper Validation and Dimensionality Reduction

Phase 2 revealed two problems: (1) the inverted train/test ratio (30 train /
72 test) wastes labeled data and inflates CV variance, and (2) 128 raw FastRP
dimensions on 30-82 rows is a curse-of-dimensionality problem that drowns the
graph signal in noise. Phase 2.5 fixes both.

**Goal:** Determine whether graph features add predictive value when given a
fair shot — proper train/test ratio and compressed dimensionality.

#### 2.5a: Configurable train/test split

Replace the `create_holdout()` approach (null-out labels, train on 30) with a
standard sklearn `train_test_split` inside `train_sklearn_classifier()`.

**What to change:**

- **`train_sklearn_classifier()`** — add `test_size: float | None = None`
  parameter:
  - `None` (default, current behavior) — train on all labeled rows, no
    internal test split. Used when the caller manages holdout externally.
  - `0.2` — 80/20 stratified train/test split. The function trains on ~82
    rows, evaluates on ~20, logs test metrics to MLflow.
- **`GDSFastRPConfig`** (and community/baseline equivalents) — add
  `test_size: float | None` field, read from `TEST_SIZE` env var.
- **`.env.example`** — add `TEST_SIZE=0.2` with comment.
- **Callers** (`gds_fastrp_features.py`, `gds_community_features.py`,
  `gds_baseline_comparison.py`) — pass `test_size=cfg.test_size` through
  to `train_sklearn_classifier()`.

**When `test_size` is set:**

- Skip `create_holdout()` — all 102 rows stay labeled.
- `train_sklearn_classifier()` does `train_test_split(X, y,
  test_size=test_size, stratify=y, random_state=42)` internally.
- CV uses k=5 on the ~82 training rows (enough samples per fold now).
- After selecting the best model, evaluate on the held-out ~20 rows and
  log test F1, test accuracy, and a confusion matrix to MLflow.
- The best model is refit on the full ~82 training rows before returning.

**When `test_size` is None (default):**

- Current behavior preserved — `create_holdout()` runs externally, training
  uses all non-null rows, CV uses k=3.

#### 2.5b: PCA dimensionality reduction on FastRP

Add an optional PCA step to the sklearn Pipeline that compresses 128 FastRP
dimensions into a small number of components before training.

**What to change:**

- **`train_sklearn_classifier()`** — add `pca_components: int | None = None`
  parameter:
  - `None` (default) — no PCA, current behavior.
  - `5` — insert a PCA step into the Pipeline after StandardScaler,
    applied only to FastRP columns (identified by prefix `fastrp_`).
    Tabular features (annual_income, credit_score) pass through untouched.
- **`GDSFastRPConfig`** — add `pca_components: int | None` field, read from
  `PCA_COMPONENTS` env var.
- **`.env.example`** — add `PCA_COMPONENTS=5` with comment.
- **Callers** — pass `pca_components=cfg.pca_components` through.

**Implementation approach:**

Use `sklearn.compose.ColumnTransformer` to apply PCA only to FastRP columns:

```python
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

fastrp_cols = [c for c in feature_cols if c.startswith("fastrp_")]
other_cols = [c for c in feature_cols if not c.startswith("fastrp_")]

preprocessor = ColumnTransformer([
    ("fastrp_pca", Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_components)),
    ]), fastrp_cols),
    ("passthrough", StandardScaler(), other_cols),
])

pipe = Pipeline([("preprocess", preprocessor), ("clf", clf)])
```

This gives the model: `annual_income` + `credit_score` + 5 PCA components
= 7 features total. A much healthier ratio against 82 training rows.

#### 2.5c: Per-class metrics and confusion matrix

Log richer evaluation artifacts so we can see which classes the model
struggles with.

**What to change:**

- **`train_sklearn_classifier()`** — after selecting the best model and
  evaluating on the test set (when `test_size` is set):
  - Log `sklearn.metrics.classification_report` as a dict to MLflow params
    or as a JSON artifact.
  - Log confusion matrix via `mlflow.log_figure()` using
    `sklearn.metrics.ConfusionMatrixDisplay`.
  - Log per-class F1, precision, recall as individual MLflow metrics
    (e.g., `test_f1_Aggressive`, `test_precision_Conservative`).

#### 2.5d: Log holdout accuracy to MLflow

- **`evaluate_predictions()`** — add optional `run_id: str | None` parameter.
  When set, log the held-out accuracy as `mlflow.log_metric("holdout_accuracy",
  accuracy)` on the given run. This makes cross-experiment comparison possible
  in the MLflow UI.

#### Expected experiment matrix

Run all three experiments with Phase 2.5 settings and compare against
Phase 1/2 baselines:

| Experiment | Phase 1-2 F1 | Phase 2.5 Expected | Why |
|------------|-------------|-------------------|-----|
| Tabular only | 0.853 | ~0.85-0.90 | More training data, same 2 features |
| FastRP + PCA(5) | 0.636 | **~0.80-0.85?** | PCA compresses graph signal, fair feature ratio |
| FastRP + Louvain + PCA(5) | 0.636 | ~0.80-0.85? | Community_id may still add nothing |
| FastRP raw (128 dims) | 0.636 | ~0.70-0.75? | More training data helps, but still too many dims |

The key comparison is **Tabular only vs FastRP + PCA(5)**. If PCA-compressed
graph features close the gap or beat tabular-only, graph enrichment adds value.
If not, the tabular model is the right Champion.

#### Files that change

| File | What Changes |
|------|-------------|
| `src/graph_feature_forge/ml/automl_training.py` | Add `test_size`, `pca_components` params; ColumnTransformer for PCA; per-class metrics; confusion matrix logging |
| `agent_modules/gds_fastrp_features.py` | Read `TEST_SIZE`/`PCA_COMPONENTS` from env, pass through; conditionally skip `create_holdout()` when `test_size` is set |
| `agent_modules/gds_community_features.py` | Same |
| `agent_modules/gds_baseline_comparison.py` | Same |
| `.env.example` | Add `TEST_SIZE=0.2` and `PCA_COMPONENTS=5` |
| `pyproject.toml` | Version bump to 0.5.0 |

### Phase 2.5: Todo

- [x] Add `test_size` parameter to `train_sklearn_classifier()`
  - [x] `train_test_split` with stratification when `test_size` is set
  - [x] Bump CV to k=5 when training set is large enough
  - [x] Evaluate on internal test set, log test F1 and accuracy to MLflow
  - [x] Refit best model on full training set before returning
- [x] Add `pca_components` parameter to `train_sklearn_classifier()`
  - [x] `ColumnTransformer` with PCA on `fastrp_*` columns, passthrough for others
  - [x] Log `pca_components` and `explained_variance_ratio` to MLflow
- [x] Add per-class metrics logging
  - [x] Per-class F1/precision/recall as individual MLflow metrics
  - [x] Confusion matrix plot via `mlflow.log_figure()`
- [x] Add `run_id` parameter to `evaluate_predictions()` for MLflow logging
- [x] Add `TEST_SIZE` and `PCA_COMPONENTS` to config classes and `.env.example`
- [x] Update callers to pass new params and conditionally skip `create_holdout()`
- [x] Bump version to 0.5.0 (then 0.5.1 to fix kNN concurrency bug)
- [x] Build wheel and upload
- [x] Run experiment matrix: tabular-only, FastRP+PCA(5), FastRP+Louvain+PCA(5)
- [x] Compare results against Phase 1-2 baselines in MLflow
- [x] Document findings: does graph enrichment add value with PCA?

### Phase 2.5: Results

#### Three-way experiment comparison (80/20 split, PCA(5), 5-fold CV)

| Experiment | Best cv_F1 | Best test_F1 | Best Model | Features |
|------------|-----------|-------------|------------|----------|
| **FastRP only + PCA(5)** | **0.8010** | — | GradientBoosting | 7 (5 PCA + 2 tabular) |
| FastRP + Louvain + PCA(5) | 0.7891 | 0.9028 | GradientBoosting | 8 (5 PCA + 3 tabular) |
| Tabular only | 0.7839 | 0.8519 | RandomForest | 2 (annual_income, credit_score) |

#### Comparison to Phase 1-2 (30 train rows, no PCA)

| Experiment | Phase 1-2 F1 | Phase 2.5 F1 | Change |
|------------|-------------|-------------|--------|
| FastRP only | 0.636 | **0.801** | +0.165 |
| FastRP + Louvain | 0.636 | 0.789 | +0.153 |
| Tabular only | 0.853 | 0.784 | -0.069 |

#### FastRP + Louvain + PCA(5) detail

| Classifier | cv_F1 | CV Std | test_F1 |
|------------|-------|--------|---------|
| **GradientBoosting** | **0.7891** | 0.0906 | **0.9028** |
| RandomForest | 0.7469 | 0.0950 | 0.9028 |
| LogisticRegression | 0.6954 | 0.0527 | 0.7000 |

#### Tabular-only detail

| Classifier | cv_F1 | CV Std | test_F1 |
|------------|-------|--------|---------|
| **RandomForest** | **0.7839** | 0.0662 | 0.8519 |
| LogisticRegression | 0.7627 | 0.0549 | 0.7204 |
| GradientBoosting | 0.7591 | 0.0926 | 0.9070 |

#### Feature importance (FastRP + Louvain + PCA(5) model)

Features 0-4 are PCA-compressed graph structure (FastRP embeddings reduced
from 128 dims to 5 components). Features 5-6 are annual_income and
credit_score. Feature 7 is community_id.

| Feature | Importance | What it is |
|---------|-----------|------------|
| 5 (annual_income) | 48.7% | Tabular |
| 6 (credit_score) | 38.2% | Tabular |
| 0 (PCA component 0) | 3.8% | Graph structure |
| 7 (community_id) | 2.8% | Graph structure |
| 2 (PCA component 2) | 2.1% | Graph structure |
| 1 (PCA component 1) | 1.9% | Graph structure |
| 4 (PCA component 4) | 1.4% | Graph structure |
| 3 (PCA component 3) | 1.0% | Graph structure |

Graph features contribute ~13% of total importance. Income and credit
score remain dominant (~87%), but the graph signal adds a real secondary
signal that improves cv_F1 from 0.784 (tabular-only) to 0.801 (FastRP+PCA).

**Note:** `extract_feature_importance()` reports "0/20 graph features" because
after PCA the features are unnamed integer indices. The function checks for
names starting with `fastrp_` or `community_id`, which no longer match.
This is a cosmetic reporting bug, not a model issue.

#### Key Findings

1. **Graph features now add value.** FastRP+PCA(5) cv_F1=0.801 beats
   tabular-only cv_F1=0.784. This reverses the Phase 2 finding where
   tabular-only (0.853) crushed FastRP (0.636).

2. **PCA was the critical fix.** Compressing 128 FastRP dims to 5 components
   eliminated the curse-of-dimensionality problem. The feature-to-sample
   ratio went from 130:30 (4.3:1, noise-dominated) to 7:81 (0.09:1, healthy).

3. **More training data helped.** 81 train rows vs 30 made CV estimates
   more stable — CV std dropped from 0.19 to 0.09.

4. **Community_id contributes 2.8%.** Slightly useful as a feature, but
   Louvain community detection doesn't add much over FastRP alone
   (cv_F1=0.801 vs 0.789).

5. **Test F1 of 0.903 is strong.** The model generalizes well on 21
   held-out rows, though small test sets have high variance.

6. **Tabular-only cv_F1 dropped from 0.853 to 0.784.** The Phase 1-2
   result (0.853) was likely overfit — training on only 30 rows with 2
   features and k=3 CV gave an optimistic estimate. The Phase 2.5 setup
   (81 rows, k=5 CV) is more reliable.

#### Bug fixed during Phase 2.5

GDS 2026.3.0 requires `concurrency=1` when `randomSeed` is set on
`gds.knn.mutate()`. Without it, the procedure raises
`IllegalArgumentException`. Fixed in v0.5.1 by adding `concurrency=1`
to the kNN call in `run_knn_analysis()`.

### Phase 3: Cleanup and Research

#### 3a: Code cleanup

- [ ] Remove old `train_automl_classifier()` — the sklearn path is stable
  and outperforms AutoML. Dead code.
- [ ] Fix `extract_feature_importance()` to handle PCA feature names —
  after PCA, features are unnamed integer indices. The function should
  map indices back to meaningful labels (e.g. "PCA_0", "annual_income")
  when a `ColumnTransformer` is detected in the pipeline.

#### 3b: PCA component sweep

Determine whether PCA(5) is optimal or if fewer/more components capture
the graph signal better.

**What to do:**

- Run the FastRP+PCA pipeline with `PCA_COMPONENTS=3`, `5`, `10`, `15`
  (four runs, same `TEST_SIZE=0.2`)
- Compare cv_F1 and test_F1 across runs in MLflow
- Check `pca_explained_variance_total` — how much of the FastRP variance
  is captured at each level?
- Pick the component count that maximizes cv_F1 without overfitting
  (test_F1 should not diverge too far from cv_F1)

**Expected outcome:** PCA(3) may lose too much signal. PCA(10) may
reintroduce noise. PCA(5) is likely near-optimal but worth verifying.

#### 3c: Isolate PCA vs more data

Phase 2.5 changed two variables at once (PCA + more training data).
Isolate which mattered more.

**What to do:**

- Run FastRP raw (128 dims, no PCA) with 80/20 split:
  set `TEST_SIZE=0.2`, unset `PCA_COMPONENTS`
- Compare against Phase 2.5 FastRP+PCA(5) (cv_F1=0.801) and
  Phase 1-2 FastRP raw (cv_F1=0.636)

**Possible outcomes:**

| FastRP raw + 80/20 F1 | Interpretation |
|----------------------|----------------|
| ~0.80 (matches PCA) | More data was the fix, PCA is optional |
| ~0.70 (between) | Both helped, PCA adds incremental value |
| ~0.64 (matches Phase 1-2) | PCA was the critical fix, more data alone doesn't help |

This determines whether PCA should be the default or an optional
enhancement.

#### Phase 3: Todo

- [ ] Remove `train_automl_classifier()` from `automl_training.py`
- [ ] Fix `extract_feature_importance()` for PCA pipelines
- [ ] Run PCA sweep: PCA(3), PCA(5), PCA(10), PCA(15)
- [ ] Run FastRP raw (no PCA) with 80/20 split
- [ ] Document findings: optimal PCA components and PCA vs data contribution

---

## Reference Material

These are the Databricks docs and notebooks that informed this plan:

- **Databricks sklearn docs:** https://docs.databricks.com/aws/en/machine-learning/train-model/scikit-learn
- **sklearn classification notebook (Unity Catalog):** https://docs.databricks.com/aws/en/notebooks/source/machine-learning-with-unity-catalog.html
- **Hyperopt model selection notebook:** https://docs.databricks.com/aws/en/notebooks/source/hyperopt-sklearn-model-selection.html
- **Optuna + MLflow 3.0 docs:** https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/optuna
- **Optuna reference notebook:** https://docs.databricks.com/aws/en/notebooks/source/machine-learning/optuna-mlflow.html
- **ai-dev-kit classical ML patterns:** `databricks-skills/databricks-model-serving/1-classical-ml.md`
- **ai-dev-kit ML training pipeline example:** `databricks-skills/databricks-jobs/examples.md` (lines 337-438)
