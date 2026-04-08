# GDS Pipeline — Status and Known Issues

**Date:** 2026-04-08

## Current Status

### Run in Progress

Run ID `288135748309650` is executing `gds_fastrp_features.py` on cluster `0408-142636-b62ttvaa`.
Previous runs confirmed Steps 1-3 pass. The run timed out locally (20-min default) but is still
running on Databricks — likely in Step 4 (AutoML training, which can take 20-30 min).

| Step | Description | Status |
|------|-------------|--------|
| 1/5  | Compute GDS features (FastRP + Louvain) | PASS — 539 nodes, 890 rels, 136 communities |
| 2/5  | Export feature table to Delta | PASS (after Spark Connector fix) — 102 rows, 132 cols |
| 3/5  | Create holdout split | PASS — 30 labels kept, 72 held out |
| 4/5  | Train AutoML classifier | IN PROGRESS |
| 5/5  | Score held-out + writeback to Neo4j | PENDING |

After `gds_fastrp_features.py` completes, `run_pipeline.sh gds` runs two more jobs sequentially:
- `gds_community_features.py` — adds Louvain community_id, retrains, promotes if F1 improves
- `gds_baseline_comparison.py` — tabular-only baseline + 3-way MLflow comparison

## Cluster Configuration

### Current Cluster

| Setting | Value |
|---------|-------|
| Cluster ID | `0408-142636-b62ttvaa` |
| Name | GDS Feature Forge (16.4 LTS ML) |
| Runtime | **16.4.x-cpu-ml-scala2.12** (Spark 3.5.2, Scala 2.12) |
| Mode | Single User (Unity Catalog enabled) |
| Workers | 0 (single-node) |
| Auto-termination | 120 min |

### Required Libraries (installed on cluster)

| Library | Coordinates | Why |
|---------|-------------|-----|
| Wheel | `/Volumes/graph-feature-forge/enrichment/source-data/wheels/graph_feature_forge-0.2.3-py3-none-any.whl` | Project library |
| Maven | `org.neo4j:neo4j-connector-apache-spark_2.12:5.3.2_for_spark_3` | Read/write Neo4j from Spark |

### Why This Runtime

The runtime must satisfy three constraints simultaneously:

1. **Databricks AutoML** (`databricks.automl.classify`) — removed in Runtime 18.0+, so must be <= 17.x
2. **Neo4j Spark Connector** — only published for Spark 3.x with Scala 2.12; the `_for_spark_4.0` artifact doesn't work
3. **ML Runtime** — AutoML requires the ML variant (suffix `-ml-`)

**DBR 17.3 LTS ML** (Spark 4.0, Scala 2.13) has AutoML but breaks the Neo4j Spark Connector due to Scala mismatch.  
**DBR 16.4 LTS ML** (Spark 3.5.2, Scala 2.12) satisfies all three constraints.

### Long-Term Considerations

- When Neo4j publishes a Spark 4.0 / Scala 2.13 connector, DBR 17.3 LTS ML can be used
- When Databricks deprecates AutoML in all runtimes, switch to direct MLflow or custom training
- The `graphdatascience` Python library (GDS client) is runtime-agnostic — no Spark/Scala dependency

## Known Issues to Fix

### 1. Notebooks: Missing `from_json` Parsing for Embeddings

**Files:** `notebooks/gds_fastrp_features.py` (line 174-180), `notebooks/gds_community_features.py` (line 172-173)

The Neo4j Spark Connector sometimes returns `fastrp_embedding` as a STRING column instead of ARRAY<DOUBLE>. The agent_modules entry points handle this with type detection:

```python
from pyspark.sql.types import ArrayType, DoubleType
embedding_col = customers_df.select("fastrp_embedding").schema[0].dataType
if isinstance(embedding_col, ArrayType):
    parsed_embedding = F.col("fastrp_embedding")
else:
    parsed_embedding = F.from_json(F.col("fastrp_embedding"), ArrayType(DoubleType()))
```

The notebooks still use the raw `.getItem(i)` pattern which will fail if the column is STRING type. Apply the same detection logic.

### 2. `run_pipeline.sh gds` Timeout

The `databricks-job-runner` has a 20-minute default timeout for `waiter.result()`. AutoML training
can take 30+ minutes. Options:
- Add `--no-wait` flag to each submit and poll separately
- Increase the timeout in `databricks-job-runner`
- Run each job with `--no-wait` and use `uv run python -m cli logs <run_id>` to check

### 3. `gds_community_features.py` — `is_held_out` Column Dependency

The `reapply_holdout()` function reads `is_held_out` from the ground truth table. If `gds_fastrp_features.py` is re-run and the holdout changes, the ground truth table is overwritten with the new split. This is correct behavior, but `gds_community_features.py` must always run **after** `gds_fastrp_features.py` in the same session. The `run_pipeline.sh gds` phase enforces this ordering.

### 4. Version Bump Required for Wheel Changes

The cluster caches installed wheels by name+version. After any code change to `src/graph_feature_forge/`, you must:
1. Bump the version in `pyproject.toml`
2. Rebuild and upload: `uv run python -m cli upload --wheel`
3. Restart the cluster (or it will use the cached old wheel)

Currently at version **0.2.3**.

### 5. Enrichment Log Table Not Yet Created

Step 1 logs `No enrichment log found` — this is expected on first run (the enrichment pipeline hasn't been run against this catalog). Not a bug, but means no enrichment relationship types are included in the GDS projection.

## File Inventory

### Entry Points (`agent_modules/`)

| File | Steps | Depends On |
|------|-------|------------|
| `gds_fastrp_features.py` | GDS → Export → Holdout → AutoML → Score | Neo4j, Spark Connector |
| `gds_community_features.py` | GDS → Export → Reapply holdout → Retrain → kNN | Neo4j, Spark Connector, gds_fastrp_features results |
| `gds_baseline_comparison.py` | Tabular AutoML → 3-way comparison → Feature importance | Delta tables only (no Neo4j) |

### Library Modules (`src/graph_feature_forge/`)

| File | Purpose |
|------|---------|
| `automl_training.py` | Shared AutoML functions: holdout, training, registration, evaluation, kNN, comparison |
| `feature_engineering.py` | GDS compute + export (FastRP, Louvain, feature table) |
| `extraction.py` | `spark_neo4j_options()` helper for Spark Connector config |

### Notebooks (`notebooks/`)

Original interactive notebooks. Need the `from_json` embedding fix (issue #1 above).
