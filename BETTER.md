# Proposal: Neo4j Vector Search and Classic Compute

The current pipeline runs entirely on Databricks serverless. Document retrieval happens in-memory against 20 pre-computed embedding chunks loaded from a JSON file. Structured data moves between Neo4j and Delta tables through SQL and the Neo4j Python driver. Both of these work, but they leave capability on the table. The embeddings already live in Neo4j as a vector index; the pipeline ignores it and reimplements cosine similarity in Python. The Neo4j Spark Connector exists but only runs on classic compute, which the pipeline currently avoids.

This proposal covers two changes that are independent but complementary: replacing in-memory retrieval with Neo4j's vector index, and introducing classic compute for the steps that benefit from the Spark Connector.

---

## 1. Neo4j Vector Index for Document Retrieval

### The current state

`seed_neo4j.py` already writes 1024-dimensional embeddings to every Chunk node and creates a vector index with cosine similarity:

```cypher
CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1024,
  `vector.similarity_function`: 'cosine'
}}
```

The pipeline then ignores this index entirely. `DocumentRetrieval` in `retrieval.py` loads the same embeddings from a JSON file on a UC volume, holds them in a Python list, and computes cosine similarity in a loop. This made sense as a starting point. The corpus is 20 chunks; the search completes instantly. But it means the pipeline maintains two copies of the same data (JSON file and Neo4j), the retrieval layer cannot scale beyond what fits in memory, and query results cannot incorporate graph structure.

### What changes

Replace `DocumentRetrieval` with a `Neo4jRetrieval` class that queries the existing vector index using Cypher's `db.index.vector.queryNodes` procedure. The embedder function stays the same; only the search backend changes.

The query would look like:

```cypher
CALL db.index.vector.queryNodes(
  'chunk_embedding_index',
  $top_k,
  $query_embedding
)
YIELD node, score
RETURN node.chunk_id AS chunk_id,
       node.text AS text,
       score,
       node.document_id AS document_id,
       node.document_title AS document_title,
       node.document_type AS document_type
ORDER BY score DESC
```

The `Neo4jRetrieval` class takes a Neo4j driver and an embedder, matching the same interface as the current class. `query()` embeds the text, sends the Cypher query, and returns `RetrievedChunk` objects. `format_context()` works identically.

### Graph-aware retrieval

Once retrieval runs inside Neo4j, queries can follow relationships outward from the matched chunks. A chunk that `FROM_DOCUMENT` links to a Document that `DESCRIBES` a Customer gives the synthesis step a direct path from unstructured evidence to the entity it concerns. The current in-memory approach cannot do this; it returns chunks in isolation and leaves the LLM to infer which customer a chunk relates to.

A graph-aware query could traverse one hop from the matched chunks:

```cypher
CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $embedding)
YIELD node AS chunk, score
OPTIONAL MATCH (chunk)-[:FROM_DOCUMENT]->(doc:Document)-[:DESCRIBES]->(c:Customer)
RETURN chunk.chunk_id AS chunk_id,
       chunk.text AS text,
       score,
       doc.document_id AS document_id,
       doc.title AS document_title,
       doc.document_type AS document_type,
       c.customer_id AS customer_id,
       c.first_name + ' ' + c.last_name AS customer_name
ORDER BY score DESC
```

This returns the customer identity alongside the text, eliminating a class of errors where the LLM misattributes a profile excerpt to the wrong customer.

### What this removes

The `data/embeddings/document_chunks_embedded.json` file and the `upload --data` step that pushes it to the UC volume become unnecessary for retrieval. The seeding step already writes the same data to Neo4j. The JSON file could be retained for reproducibility or as a fallback, but the retrieval hot path no longer depends on it.

### Implementation

- Add `Neo4jRetrieval` class to `retrieval.py` (or a new `neo4j_retrieval.py` if the module gets too large)
- Reuse the existing `make_sdk_embedder()` for query-time embedding
- Accept Neo4j connection parameters from the existing `Config` dataclass (already has `neo4j_uri`, `neo4j_username`, `neo4j_password`)
- Wire into `run_semantic_auth.py` step 6, replacing `DocumentRetrieval.from_json_path()`
- Keep `DocumentRetrieval` as a fallback for environments without Neo4j access

### Tradeoffs

The pipeline gains a hard dependency on Neo4j being available during the enrichment run, not just during seeding. Today, `seed_neo4j.py` writes to Neo4j once and the rest of the pipeline can run with only Delta tables and the JSON file. With Neo4j retrieval, the enrichment step needs a live Neo4j connection for document search. This is acceptable because `run_semantic_auth.py` already requires Neo4j for extraction (step 2) and write-back (step 11).

Latency is a non-concern at this scale. A vector index query over 20 chunks returns in single-digit milliseconds. At thousands or tens of thousands of chunks, the vector index would be faster than the in-memory approach since Neo4j uses HNSW approximate nearest neighbor search rather than brute-force comparison.

---

## 2. Classic Compute for Spark Connector Operations

### The current state

The pipeline has two steps that move data between Neo4j and Databricks:

1. **Loading** (`load_data.py`): reads CSVs from a UC volume and creates Delta tables via `read_files()` SQL. Runs on serverless.
2. **Extraction** (`run_semantic_auth.py` step 2): reads the full graph from Neo4j via the Neo4j Spark Connector and writes Delta tables. This step is defined in `extraction.py` but requires `org.neo4j.spark.DataSource`, a Spark JAR that is not available on serverless.

The extraction step currently fails on serverless because the connector JAR cannot be installed there. The pipeline works around this with `--skip-extraction`, relying on the Delta tables created by `load_data.py` instead. This means the enrichment pipeline never reads the live graph state; it reads the CSV-derived tables, which are a snapshot of the original workshop data.

For seeding (`seed_neo4j.py`), the pipeline uses the Neo4j Python driver directly, writing nodes and relationships via UNWIND batches over the Bolt protocol. This works on serverless because it has no Spark dependency. But it is serial, single-threaded, and cannot parallelize writes across partitions.

### What changes

Introduce a classic compute cluster for the two steps that benefit from the Spark Connector, while keeping the remaining steps on serverless where they belong.

**Step 1: Seeding via Spark Connector (replaces Python driver writes)**

Instead of reading Delta tables row by row and writing UNWIND batches through the Python driver, use the Spark Connector's write path:

```python
df = spark.read.table(f"{catalog}.{schema}.customer")
(df.write
   .format("org.neo4j.spark.DataSource")
   .option("url", neo4j_uri)
   .option("labels", ":Customer")
   .option("node.keys", "customer_id")
   .mode("overwrite")
   .save())
```

The connector handles batching, parallelism, and retries. For 800 nodes this difference is marginal, but the code is simpler and the approach scales linearly with partition count.

Relationship writes use the connector's keys mode:

```python
df = spark.read.table(f"{catalog}.{schema}.has_account")
(df.write
   .format("org.neo4j.spark.DataSource")
   .option("url", neo4j_uri)
   .option("relationship", "HAS_ACCOUNT")
   .option("relationship.save.strategy", "keys")
   .option("relationship.source.labels", ":Customer")
   .option("relationship.source.node.keys", "source.customer_id:customer_id")
   .option("relationship.target.labels", ":Account")
   .option("relationship.target.node.keys", "target.account_id:account_id")
   .mode("overwrite")
   .save())
```

**Step 2: Extraction with live graph state**

`extraction.py` already implements this correctly. It uses `org.neo4j.spark.DataSource` to read nodes and relationships in parallel and writes Delta tables. The only reason it does not run today is the serverless JAR limitation. On classic compute with the connector JAR installed, it works without code changes.

### Compute strategy

The `databricks-job-runner` already supports both `serverless` and `cluster` compute modes via `DATABRICKS_COMPUTE_MODE` in `.env`. The pipeline would use:

| Step | Compute | Reason |
|------|---------|--------|
| `load_data.py` | serverless | Pure SQL, no external JARs needed |
| `seed_neo4j.py` | classic | Neo4j Spark Connector JAR required for write path |
| `run_semantic_auth.py` (extraction) | classic | Neo4j Spark Connector JAR required for read path |
| `run_semantic_auth.py` (synthesis + analysis) | serverless | LLM calls and DSPy, no Spark dependency |

This raises a question: should extraction be a separate script from enrichment? Currently `run_semantic_auth.py` does both (extract in step 2, then synthesize/analyze in steps 4-11). Splitting extraction into its own `extract_graph.py` script would let each step run on its optimal compute. The enrichment steps have no Spark Connector dependency and benefit from serverless cold-start speed and cost model.

### Cluster configuration

The classic cluster needs the Neo4j Spark Connector JAR. The recommended approach is a single-node cluster with the JAR installed via init script or cluster library:

```
Maven coordinates: org.neo4j:neo4j-connector-apache-spark_2.12:5.3.1_for_spark_3
```

The cluster can auto-terminate after idle timeout since these jobs run infrequently. The `databricks-job-runner` handles cluster startup automatically when `DATABRICKS_COMPUTE_MODE=cluster` and `DATABRICKS_CLUSTER_ID` is set.

### Tradeoffs

Classic compute introduces cold-start latency (1-3 minutes for cluster startup) and ongoing cost for the cluster while it is running. For the current data volume (800 nodes, 850 relationships), the Python driver's UNWIND approach completes in 30 seconds on serverless. The Spark Connector path on a classic cluster would take longer wall-clock time due to startup, even though the actual data transfer is faster.

The payoff comes at larger scale and from correctness. The extraction step reading live graph state means the enrichment pipeline operates on what Neo4j actually contains, not a stale CSV snapshot. And if the graph grows to tens of thousands of nodes, the Spark Connector's partition-level parallelism becomes meaningful.

A middle path: keep the Python driver for seeding (it works, it is fast enough, and it runs on serverless) but use classic compute only for extraction. This gets the correctness benefit of live graph reads without introducing a new seeding implementation.
