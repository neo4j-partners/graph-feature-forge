# Neo4j Spark Connector — Issue Log

**Date:** 2026-04-08

## Problem

The GDS feature engineering pipeline (`gds_fastrp_features.py`) hangs indefinitely at Step 2 ("Exporting feature table") when using the Neo4j Spark Connector to read Customer nodes from Neo4j Aura. The Neo4j Python driver works fine on the same cluster and same Neo4j instance.

## Environment

- **Databricks workspace:** Azure East US (`adb-1098933906466604.4.azuredatabricks.net`)
- **Neo4j Aura:** `neo4j+s://3eaa1d7c.databases.neo4j.io`
- **Neo4j database:** `neo4j` (102 Customer nodes)
- **Cluster ID:** `0408-142636-b62ttvaa`
- **Current runtime:** 16.4.x-cpu-ml-scala2.12 (Spark 3.5.2, Scala 2.12)
- **Connector:** `org.neo4j:neo4j-connector-apache-spark_2.12:5.3.2_for_spark_3`
- **Wheel:** `graph_feature_forge-0.2.3-py3-none-any.whl`

## What Works

- **Neo4j Python driver** (`neo4j` package): Connects, queries, returns 102 customers instantly
- **GDS Python client** (`graphdatascience`): Connects, projects graph, runs FastRP + Louvain, writes properties back to Neo4j — all in ~30 seconds
- **Spark Connector `.load()`**: Returns successfully with schema metadata (column names and types)

## What Hangs

- **Spark Connector `.count()`**: Hangs indefinitely after `.load()` succeeds
- **Spark Connector `.collect()`**: Same behavior (implicit in `write.saveAsTable()`)
- Any operation that triggers actual data read from Neo4j through the Spark Connector

## Connector Configuration

```python
from graph_feature_forge.graph.extraction import spark_neo4j_options

options = spark_neo4j_options(uri, username, password, database)
# Returns:
# {
#     "url": "neo4j+s://3eaa1d7c.databases.neo4j.io",
#     "authentication.type": "basic",
#     "authentication.basic.username": "neo4j",
#     "authentication.basic.password": "...",
#     "database": "neo4j",
# }
options["labels"] = ":Customer"

df = spark.read.format("org.neo4j.spark.DataSource").options(**options).load()
# ^ This returns with schema, but any action (count, collect, show) hangs
```

## Runtime Compatibility Matrix

Every combination was tested. None produced a working Spark Connector data read.

| Runtime | Spark | Scala | Connector Artifact | Result |
|---------|-------|-------|--------------------|--------|
| 18.1 (non-LTS) | 4.0 | 2.13 | `_2.13:5.3.2_for_spark_4.0` | `ImportError: cannot import 'automl'` (AutoML removed in 18.0) |
| 17.3 LTS ML | 4.0 | 2.13 | `_2.13:5.3.2_for_spark_4.0` | Maven install status: **FAILED** (artifact may not exist or has dependency conflicts) |
| 17.3 LTS ML | 4.0 | 2.13 | `_2.12:5.3.2_for_spark_3` | `NoClassDefFoundError: scala/Serializable` (Scala 2.12 jar on 2.13 runtime) |
| **16.4 LTS ML** | **3.5.2** | **2.12** | **`_2.12:5.3.2_for_spark_3`** | **Installs OK, schema loads, data read hangs indefinitely** |

## Diagnostic Test Script

Created `agent_modules/test_spark_connector.py` to isolate the issue:

```python
"""Quick test: does the Neo4j Spark Connector work on this cluster?"""
import os, sys, signal

def timeout_handler(signum, frame):
    print("TIMEOUT: Spark Connector read hung after 60 seconds")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)

def main():
    from graph_feature_forge import inject_params
    inject_params()

    from databricks.sdk import WorkspaceClient
    wc = WorkspaceClient()
    print(f"Connected to {wc.config.host}")

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_pass = os.getenv("NEO4J_PASSWORD")
    neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")

    # Test 1: Python driver (works)
    print("--- Test 1: Neo4j Python driver ---")
    import neo4j
    driver = neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    with driver.session(database=neo4j_db) as session:
        count = session.run("MATCH (c:Customer) RETURN count(c) AS cnt").single()["cnt"]
        print(f"  Customer count via Python driver: {count}")
    driver.close()

    # Test 2: Spark Connector (hangs on data read)
    print("--- Test 2: Neo4j Spark Connector ---")
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

    signal.alarm(60)  # Note: does NOT fire inside JVM threads
    df = (
        spark.read.format("org.neo4j.spark.DataSource")
        .option("url", neo4j_uri)
        .option("authentication.type", "basic")
        .option("authentication.basic.username", neo4j_user)
        .option("authentication.basic.password", neo4j_pass)
        .option("database", neo4j_db)
        .option("labels", ":Customer")
        .load()
    )
    signal.alarm(0)
    print(f"  Schema loaded: {df.columns[:5]}...")  # <-- reaches here
    print(f"  Row count: {df.count()}")              # <-- hangs here forever
    print("  SUCCESS")

if __name__ == "__main__":
    main()
```

### Test Results

```
Connected to https://eastus-c3.azuredatabricks.net
Neo4j URI: neo4j+s://3eaa1d7c.databases.neo4j.io

--- Test 1: Neo4j Python driver ---
  Customer count via Python driver: 102

--- Test 2: Neo4j Spark Connector ---
  Setting 60s alarm...
  Calling spark.read.format('org.neo4j.spark.DataSource').load()...
  Schema loaded: ['<id>', '<labels>', 'zip_code', 'state', 'risk_profile']...
  [HANGS INDEFINITELY — never reaches count() output]
```

**Run IDs:**
- Test script: `251600852949546` (cancelled after 5+ min hang)
- Full pipeline: `288135748309650` (cancelled after 35+ min hang at Step 2)

### Note on SIGALRM

`signal.alarm()` does not interrupt PySpark operations because the actual work happens in JVM threads, not the Python main thread. The alarm signal is only delivered when control returns to Python, which never happens when the JVM is blocked.

## Previous Runs Where Step 2 Worked

On an earlier cluster (before this session's cluster recreation), Steps 1-3 passed successfully:

```
Step 1: GDS features computed (539 nodes, 890 rels, 136 communities, modularity 0.9550)
Step 2: Feature table exported (102 rows, 132 cols)
Step 3: Holdout created (30 labels kept, 72 held out)
Step 4: FAILED — ImportError: cannot import 'automl' (Runtime 18.1)
```

That cluster was running **Runtime 18.1** (non-LTS) with the Neo4j Spark Connector for Spark 4.0. The connector worked for data reads but the runtime lacked AutoML. When we recreated the cluster on 16.4 LTS ML for AutoML support, the Spark Connector stopped working for data reads.

## Hypotheses

### 1. Connector version incompatibility with Spark 3.5.2
The `5.3.2_for_spark_3` artifact was built for Spark 3.x but may have issues specific to 3.5.2 (the latest 3.x). The connector may have been tested against Spark 3.4 or 3.3.

### 2. Network / TLS configuration difference
DBR 16.4 may have different network security settings or JVM TLS configuration than 18.1. The connector's internal Java Neo4j driver may use a different SSL context that fails silently on 16.4.

### 3. Connector deadlock on single-node cluster
With 0 workers (single-node mode), all Spark execution happens on the driver. The connector may have a threading issue or deadlock when the driver both manages the query plan and executes the Neo4j reads in the same JVM.

### 4. Aura connection pool exhaustion
Step 1 (GDS Python client) opens connections to Neo4j and may not fully close them. When the Spark Connector tries to open new connections, Aura may be throttling or queueing them. However, this seems unlikely since the schema query succeeds.

## Proposed Fix: Bypass Spark Connector for Reads

Since there are only ~102 customers, replace the Spark Connector read with:

```python
# Read via Python driver → pandas → Spark DataFrame
import neo4j
import pandas as pd

driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))
with driver.session(database=database) as session:
    result = session.run("""
        MATCH (c:Customer)
        RETURN c.customer_id AS customer_id,
               c.annual_income AS annual_income,
               c.credit_score AS credit_score,
               c.risk_profile AS risk_category,
               c.community_id AS community_id,
               c.fastrp_embedding AS fastrp_embedding
    """)
    records = [dict(r) for r in result]
driver.close()

pdf = pd.DataFrame(records)
customers_df = spark.createDataFrame(pdf)
```

**Trade-offs:**
- **Pro:** Works on any Databricks runtime, no Maven library dependency, no Scala version constraints
- **Pro:** Eliminates the entire Spark/Scala/Connector compatibility matrix
- **Pro:** The cluster runtime choice can be driven purely by AutoML requirements
- **Con:** Doesn't scale beyond ~100K rows (irrelevant for 102 customers)
- **Con:** Loses Spark-level parallelism for reads (irrelevant for 102 rows)

The same approach applies to the Spark Connector write-back in `_score_and_evaluate()` — replace with Python driver `UNWIND` + `MERGE`.

## Affected Files

| File | Uses Spark Connector For |
|------|-------------------------|
| `agent_modules/gds_fastrp_features.py` | Read customers (export), write predictions (score) |
| `agent_modules/gds_community_features.py` | Read customers (export via `feature_engineering.export_feature_table`) |
| `src/graph_feature_forge/feature_engineering.py` | `export_feature_table()` — read customers; `score_unlabeled_customers()` — write predictions |
| `src/graph_feature_forge/extraction.py` | `extract_nodes()`, `extract_relationships()` — bulk extraction (not used by GDS pipeline) |
| `notebooks/gds_fastrp_features.py` | Read customers for feature table |
| `notebooks/gds_community_features.py` | Read customers for feature table |

The `extraction.py` functions are used by the data loading pipeline (`load_data.py`), not the GDS pipeline. Those can continue using the Spark Connector if running on a compatible runtime (18.x where it works).
