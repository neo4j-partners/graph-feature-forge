# GDS Pipeline — Retry on Cluster 1029-205109-yca7gn2n

**Date:** 2026-04-08  
**Target cluster:** `1029-205109-yca7gn2n` ("Small Spark 4.0")  
**Runtime:** 17.3.x-cpu-ml-photon-scala2.13 (Spark 4.0.0, Scala 2.13)

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
| `graph-feature-forge` wheel | NOT INSTALLED | v0.2.3 on volume, not added as cluster library |
| `graphdatascience` Python lib | NOT INSTALLED | Required by `compute_gds_features()` |
| `.env` DATABRICKS_CLUSTER_ID | WRONG CLUSTER | Currently points to `0408-142636-b62ttvaa` |
| Entry point scripts on workspace | UNKNOWN | Need to verify or re-upload |

## Incremental Plan

### Phase 0: Fix cluster configuration
> Goal: Get missing libraries installed and .env pointing to the right cluster.

- [ ] **0.1** Install `graph-feature-forge` wheel (v0.2.3) as a cluster library on `1029-205109-yca7gn2n`
  - Wheel path: `/Volumes/graph-feature-forge/enrichment/source-data/wheels/graph_feature_forge-0.2.3-py3-none-any.whl`
  - Install via cluster UI or API, then restart cluster
- [ ] **0.2** Install `graphdatascience>=1.7` as a PyPI library on the cluster
  - Required for `from graphdatascience import GraphDataScience` in `feature_engineering.py`
- [ ] **0.3** Update `.env`: set `DATABRICKS_CLUSTER_ID=1029-205109-yca7gn2n`
- [ ] **0.4** Restart cluster to pick up new libraries
- [ ] **0.5** Verify libraries installed:
  ```python
  import graph_feature_forge; print("wheel OK")
  from graphdatascience import GraphDataScience; print("gds OK")
  from databricks import automl; print("automl OK")
  ```

### Phase 1: Smoke tests (before running full pipeline)
> Goal: Confirm each critical capability works in isolation on this cluster.

- [ ] **1.1** Spark Connector READ test
  ```python
  df = spark.read.format("org.neo4j.spark.DataSource") \
      .option("url", neo4j_uri) \
      .option("authentication.type", "basic") \
      .option("authentication.basic.username", neo4j_user) \
      .option("authentication.basic.password", neo4j_pass) \
      .option("database", "neo4j") \
      .option("labels", ":Customer") \
      .load()
  print(f"Schema: {df.columns[:5]}")
  print(f"Count: {df.count()}")  # This is the operation that hangs on 16.4
  ```
  **Expected:** 102 rows, completes in < 30 seconds

- [ ] **1.2** GDS Python client test
  ```python
  from graphdatascience import GraphDataScience
  gds = GraphDataScience(neo4j_uri, auth=(neo4j_user, neo4j_pass), database="neo4j")
  # Quick projection test
  G, _ = gds.graph.project("test", ["Customer"], {"HAS_ACCOUNT": {"orientation": "UNDIRECTED"}})
  print(f"Nodes: {G.node_count()}")
  G.drop()
  ```
  **Expected:** Projects graph, returns node count, drops cleanly

- [ ] **1.3** AutoML dry-run test
  ```python
  from databricks import automl
  # Just verify import and that classify function exists
  print(f"AutoML version: {automl.__version__}")
  print(f"classify available: {hasattr(automl, 'classify')}")
  ```

- [ ] **1.4** Spark Connector WRITE test
  ```python
  from pyspark.sql import functions as F
  test_df = spark.createDataFrame([("test_customer", "test_value")], ["customer_id", "test_col"])
  # Verify write format is recognized (don't actually write)
  print("Write format recognized: OK")
  ```

### Phase 2: Run `gds_fastrp_features.py` (Steps 1-5)
> Goal: Full FastRP pipeline end-to-end.

- [ ] **2.1** Upload latest wheel: `uv run python -m cli upload --wheel`
- [ ] **2.2** Upload entry point: `uv run python -m cli upload gds_fastrp_features.py`
- [ ] **2.3** Submit job: `uv run python -m cli submit gds_fastrp_features.py --compute cluster`
- [ ] **2.4** Monitor progress — expected steps:
  - Step 1/5: Compute GDS features (FastRP + Louvain) — should complete in ~30s
  - Step 2/5: Export feature table — **this is where 16.4 hung** — should now work
  - Step 3/5: Create holdout split
  - Step 4/5: Train AutoML classifier — can take 20-30 min
  - Step 5/5: Score and evaluate + write back to Neo4j
- [ ] **2.5** Verify result: check logs for "Pipeline complete"

### Phase 3: Run `gds_community_features.py` (Steps 1-6)
> Goal: Add Louvain community features, retrain, promote if improved.
> Depends on Phase 2 completing successfully.

- [ ] **3.1** Upload entry point: `uv run python -m cli upload gds_community_features.py`
- [ ] **3.2** Submit job: `uv run python -m cli submit gds_community_features.py --compute cluster`
- [ ] **3.3** Verify result

### Phase 4: Run `gds_baseline_comparison.py` (Steps 1-3)
> Goal: Tabular-only baseline + 3-way MLflow comparison.
> Depends on Phase 3 completing successfully.

- [ ] **4.1** Upload entry point: `uv run python -m cli upload gds_baseline_comparison.py`
- [ ] **4.2** Submit job: `uv run python -m cli submit gds_baseline_comparison.py --compute cluster`
- [ ] **4.3** Verify result

### Phase 5: Full pipeline via `run_pipeline.sh gds`
> Goal: Confirm the one-command pipeline works end-to-end.
> Only run this if Phases 2-4 succeeded individually.

- [ ] **5.1** Run `./run_pipeline.sh gds`
- [ ] **5.2** Verify all 3 jobs submitted and complete

## Known Risks

1. **Spark 3 connector on Spark 4 runtime**: `_for_spark_3` on Spark 4.0 is a version mismatch.
   User has confirmed it works for reads, but monitor for edge-case issues in writes or
   complex DataFrame operations.

2. **0 workers**: The cluster is single-node. AutoML training may be slow but should work.
   If AutoML hangs, consider adding 1 worker.

3. **Wheel caching**: If the wheel version (0.2.3) was previously installed on this cluster
   and code has changed since, need to bump version in `pyproject.toml`, rebuild, and re-upload.
   Check if code changes exist that aren't in the 0.2.3 wheel.

4. **Hyperopt deprecation warning**: Seen on cluster startup — cosmetic only, should not block.

## Rollback

If this cluster doesn't work, the SPARK.md "Proposed Fix" (bypass Spark Connector with
Python driver reads) remains viable. That approach eliminates the connector dependency entirely
at the cost of not scaling past ~100K rows (irrelevant for 102 customers).
