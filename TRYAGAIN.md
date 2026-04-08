# GDS Pipeline — Retry on Cluster 1029-205109-yca7gn2n

**Date:** 2026-04-08  
**Target cluster:** `1029-205109-yca7gn2n` ("Small Spark 4.0")  
**Runtime:** 17.3.x-cpu-ml-photon-scala2.13 (Spark 4.0.0, Scala 2.13)  
**Current wheel:** v0.3.1  
**Current status:** Phase 2 Step 4 (AutoML training) — IN PROGRESS (v0.3.1)

## Why This Cluster

The 16.4 LTS ML cluster (`0408-142636-b62ttvaa`) has a confirmed Spark Connector hang:
the connector loads schema but any data action (count, collect) hangs indefinitely.
The "Warning: compute in this mode needs at least 1 worker to run Spark commands"
suggests single-node mode contributes, but 1029 is also 0-worker and the connector
works there — so the issue is runtime/connector-version-specific, not worker-count.

This cluster has everything the GDS pipeline needs:
- Neo4j Spark Connector `_2.13:5.3.10_for_spark_3` (confirmed working for reads)
- Databricks AutoML v1.31.3 (available on 17.3 LTS ML, removed in 18.0+)
- ML Runtime (required for `databricks.automl.classify`)

## Pre-flight Audit

| Requirement | Status | Notes |
|-------------|--------|-------|
| Neo4j Spark Connector JAR | INSTALLED | `org.neo4j:neo4j-connector-apache-spark_2.13:5.3.10_for_spark_3` via Maven |
| AutoML (`databricks.automl`) | AVAILABLE | v1.31.3 |
| `graph-feature-forge` wheel | PER-JOB | Attached by `databricks-job-runner` at submit time via `task.libraries` |
| `graphdatascience` Python lib | INSTALLED | v1.20 as PyPI cluster library |
| `.env` DATABRICKS_CLUSTER_ID | CORRECT | Points to `1029-205109-yca7gn2n` |
| Unity Catalog | MIGRATED | `graph_feature_forge` (underscores, v0.3.0+) |

## Incremental Plan

### Phase 0: Fix cluster configuration — DONE

- [x] **0.1** ~~Install `graph-feature-forge` wheel~~ — handled per-job by `databricks-job-runner`
- [x] **0.2** Install `graphdatascience==1.20` as a PyPI library on the cluster
- [x] **0.3** Update `.env`: set `DATABRICKS_CLUSTER_ID=1029-205109-yca7gn2n`
- [x] **0.4** Cluster running with new libraries
- [x] **0.5** Libraries verified: graphdatascience 1.20, automl 1.31.3, neo4j 6.1.0, mlflow 3.11.1

### Phase 1: Smoke tests — DONE

- [x] **1.1** Spark Connector READ — **PASS** (102 rows in 4.8s)
- [x] **1.2** GDS Python client — **PASS** (GDS v2026.3.0, 102 nodes projected)
- [x] **1.3** AutoML import — **PASS** (v1.31.3, `classify` available)

### Phase 2: Run `gds_fastrp_features.py` (Steps 1-5) — IN PROGRESS

- [x] **2.1** Upload latest wheel (v0.3.1)
- [x] **2.2** Upload entry point
- [ ] **2.3** Submit job — **IN PROGRESS (v0.3.1)**
  - Step 1/5: Compute GDS features — **PASS** (539 nodes, 890 rels, 136 communities)
  - Step 2/5: Export feature table — **PASS** (102 rows, 132 cols)
  - Step 3/5: Create holdout split — **PASS** (30 labels kept, 72 held out)
  - Step 4/5: Train AutoML classifier — **IN PROGRESS** (v0.3.1, ~20-30 min expected)
  - Step 5/5: Score and evaluate + write back to Neo4j — PENDING
- [ ] **2.4** Verify result: check logs for "Pipeline complete"

### Phase 3: Run `gds_community_features.py` — PENDING

- [ ] **3.1** Submit (runs automatically after Phase 2 in `run_pipeline.sh`)
- [ ] **3.2** Verify result

### Phase 4: Run `gds_baseline_comparison.py` — PENDING

- [ ] **4.1** Submit (runs automatically after Phase 3 in `run_pipeline.sh`)
- [ ] **4.2** Verify result

## Catalog Migration (v0.3.0)

Renamed Unity Catalog from `graph-feature-forge` to `graph_feature_forge` (underscores).
This was required because the hyphenated name caused unfixable issues with AutoML Service
mode (see Key Insights below).

**What changed:**
- Created new catalog `graph_feature_forge` with schema `enrichment` and volume `source-data`
- Updated `.env`: `CATALOG_NAME=graph_feature_forge`, `DATABRICKS_VOLUME_PATH` updated
- Removed backtick escaping from all agent module table name properties
- Old catalog `graph-feature-forge` still exists (not deleted)

## Bugs Found and Fixed

### From first batch run (all 3 jobs submitted concurrently, all failed)

| # | Version | Bug | Root Cause | Fix | File |
|---|---------|-----|------------|-----|------|
| 1 | 0.2.4 | Jobs run concurrently, not sequentially | `run_pipeline.sh` submits all 3 with `--no-wait` but they have strict dependencies | Remove `--no-wait` from first two submits | `run_pipeline.sh` |
| 2 | 0.2.4 | Delta schema mismatch when community adds `community_id` | `saveAsTable` rejects schema changes without explicit opt-in | `.option("overwriteSchema", "true")` on all `saveAsTable` calls | `ml/feature_engineering.py`, `gds_fastrp_features.py` |
| 3 | 0.2.4 | Baseline crashes if `community_id` column missing | `exclude_cols` unconditionally includes `community_id` | Filter `exclude_cols` against actual table schema | `gds_baseline_comparison.py` |

### From sequential runs (fixing AutoML issues iteratively)

| # | Version | Bug | Root Cause | Fix | File |
|---|---------|-----|------------|-----|------|
| 4 | 0.2.4 | AutoML experiment directory missing | AutoML stores experiments under `/Users/<user>/databricks_automl/`, parent dirs must exist | `ws.workspace.mkdirs()` before `automl.classify()` | `ml/automl_training.py` |
| 5 | 0.2.5 | mkdirs creates wrong path | Created `/Shared/graph-feature-forge/` but AutoML needs `/Users/<user>/databricks_automl/Shared/graph-feature-forge/` | Construct full AutoML-internal path using `ws.current_user.me().user_name` | `ml/automl_training.py` |
| 6 | 0.2.6 | AutoML double-backticks table name | Catalog `graph-feature-forge` has hyphens; AutoML's internal notebook re-escapes backticks | Renamed catalog to `graph_feature_forge` (v0.3.0) | `.env`, agent_modules |
| 7 | 0.2.6 | Stripped backticks → invalid identifier | Removing backticks leaves `graph-feature-forge` unquoted, invalid SQL | Same catalog rename | `.env`, agent_modules |
| 8 | 0.2.7 | Stale experiment from failed run | Previous run created experiment but training failed; retry gets `RESOURCE_ALREADY_EXISTS` | Manual cleanup required between retries (stale experiments + workspace notebooks) | Manual |
| 9 | 0.2.8 | AutoML rejects DataFrame with null target labels | 72/102 rows have null `risk_category` (holdout); AutoML says "not enough rows per class" | Filter `.filter(F.col(target_col).isNotNull())` before passing to AutoML | `ml/automl_training.py` |
| 10 | 0.2.8–0.3.0 | AutoML Service can't see filtered DataFrame | AutoML Service mode creates global temp view from DataFrame; internal job can't access it | Persist labeled-only rows to `automl_training_input` table, pass table name string | `ml/automl_training.py` |

### Key insight: AutoML + hyphenated catalog names

The catalog name `graph-feature-forge` (with hyphens) caused a cascade of **unfixable** issues
with `databricks.automl.classify()`:

- **String with backticks** → AutoML's internal pipeline double-escapes → `` ``graph-feature-forge`` `` (invalid)
- **String without backticks** → `spark.table()` rejects unquoted hyphens → `INVALID_IDENTIFIER`
- **DataFrame** → AutoML Service mode creates a global temp view → internal job can't see it

There is no escaping strategy that works across all layers. The **only fix** is to use a
catalog name without special characters. Renamed to `graph_feature_forge` (underscores).

### Key insight: AutoML Service mode and DataFrames

On runtimes where `is_automl_service_enabled()` is true (17.3+), AutoML converts a
DataFrame to a global temp view and passes the view name to an internal job. That job
cannot see the temp view. The workaround is to **pass a table name string** pointing to a
persisted table with only the labeled rows (no nulls).

## Current Approach (v0.3.1)

The `train_automl_classifier` function now:
1. Filters null labels from the feature table
2. Persists labeled-only rows to `graph_feature_forge.enrichment.automl_training_input`
3. Passes that table name **as a string** to `automl.classify()`

This avoids both the hyphenated catalog escaping issues (catalog renamed) and the
global temp view visibility issue (table name string, not DataFrame).

## Run History

| Run ID | Version | Steps Passed | Failed At | Error |
|--------|---------|-------------|-----------|-------|
| 203708772673628 | 0.2.3 | 1-3 | Step 4 | AutoML dir missing |
| 957827732157554 | 0.2.3 | — | Step 2 | Delta schema mismatch (concurrent) |
| 203238771524390 | 0.2.3 | — | Step 1 | `community_id` not in schema (concurrent) |
| 405197450150596 | 0.2.4 | 1-3 | Step 4 | AutoML dir still missing (wrong path) |
| 981905236605128 | 0.2.5 | 1-3 | Step 4 | Double-backtick escaping |
| 1077784636419949 | 0.2.6 | 1-3 | Step 4 | Unquoted hyphenated identifier |
| 1025395885293576 | 0.2.7 | 1-3 | Step 4 | `RESOURCE_ALREADY_EXISTS` (stale experiment) |
| 451404531675641 | 0.2.7 | 1-3 | Step 4 | Null target labels rejected by AutoML |
| 636390056422251 | 0.2.8 | 1-3 | Step 4 | AutoML Service can't see filtered temp view |
| 1033989721769301 | 0.2.9 | 1-3 | Step 4 | `RESOURCE_ALREADY_EXISTS` (stale experiment) |
| 277164107131371 | 0.2.10 | 1-3 | Step 4 | AutoML Service still reads temp view (null labels) |
| 140190442370602 | 0.2.11 | 1-3 | Step 4 | `INVALID_IDENTIFIER` (unescaped hyphens) |
| 1100694353836738 | 0.3.0 | 1-3 | Step 4 | DataFrame temp view still invisible to AutoML Service |
| *current* | 0.3.1 | 1-3? | ? | String table name + catalog rename + persisted labeled rows |

## What's Next

### If v0.3.1 Step 4 PASSES:
- Steps 5 (scoring/writeback) runs automatically
- Phase 3 (`gds_community_features.py`) runs next in `run_pipeline.sh`
- Phase 4 (`gds_baseline_comparison.py`) runs last
- Full pipeline success = "Pipeline complete" in all 3 job logs

### If v0.3.1 Step 4 FAILS:
- Check if it's a **new error** vs the same null-labels issue
- If null-labels persists: the internal job is reading from the original feature table
  instead of `automl_training_input`. Try passing table name with explicit 3-level namespace
  or increase `holdout_per_class` so the feature table itself has enough labeled rows
- If stale experiment: manual cleanup of MLflow experiment + workspace notebooks
- Nuclear option: bypass AutoML entirely — train a scikit-learn model directly on
  the pandas DataFrame (30 rows, 130 features, 3 classes). This avoids all AutoML
  Service mode issues at the cost of fewer model types explored.

## Known Risks

1. **Spark 3 connector on Spark 4 runtime**: `_for_spark_3` on Spark 4.0 is a version mismatch.
   User has confirmed it works for reads, but monitor for edge-case issues in writes or
   complex DataFrame operations.

2. **0 workers**: The cluster is single-node. AutoML training may be slow but should work.
   If AutoML hangs, consider adding 1 worker.

3. **Wheel caching**: Each code change requires a version bump in `pyproject.toml`.
   The `databricks-job-runner` attaches the wheel per-job from the volume path.

4. **Hyperopt deprecation warning**: Seen on cluster startup — cosmetic only, does not block.

5. **Stale experiments on retry**: If AutoML creates an experiment but training fails,
   the experiment must be deleted before retrying with the same name. Manual cleanup
   required: `mlflow.delete_experiment()` + `ws.workspace.delete()` on the
   `/Users/<user>/databricks_automl/Shared/graph-feature-forge/` path.

6. **Old catalog**: `graph-feature-forge` (hyphenated) still exists. Can be dropped after
   pipeline succeeds on `graph_feature_forge`.

## Rollback

If this cluster doesn't work, the SPARK.md "Proposed Fix" (bypass Spark Connector with
Python driver reads) remains viable. That approach eliminates the connector dependency entirely
at the cost of not scaling past ~100K rows (irrelevant for 102 customers).
