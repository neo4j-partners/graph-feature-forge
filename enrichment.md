# Incremental Enrichment Cycle with Shared Schema

A graph database knows exactly what a customer holds but not what a customer wants. The holdings live in structured relationships that Cypher queries traverse in milliseconds. The intent lives in profile documents stored in Unity Catalog Volumes that the graph cannot see. Bridging that gap requires an architecture that reads unstructured documents, extracts intent, and writes new relationships back into the graph, then uses those relationships to find deeper patterns on the next pass.

The graph-feature-forge pipeline does this today, but it treats each run as a clean slate. Every cycle overwrites all fourteen Delta tables with a full extraction from Neo4j. The Neo4j Spark Connector infers column types differently from the original CSV casts, enrichment adds relationship properties the base schema never anticipated, and the pipeline fails with schema mismatches before the agents ever run. The architecture needs two things it currently lacks: a shared schema that both the lakehouse and graph agree on, and an incremental cycle that builds on prior enrichment rather than discarding it.

## The Enrichment Cycle

Each run follows a loop where the graph's current state feeds the agents, and the agents' discoveries feed the graph.

```
Ensure base tables exist (CSV → Delta, idempotent)
    ↓
Extract new graph state from Neo4j → Delta
    ↓
Agents read base + enrichment tables alongside UC Volume documents
    ↓
Propose new relationships and nodes
    ↓
Dual-write proposals to Neo4j and the enrichment log
    ↓
Repeat
```

The critical insight is that Neo4j is where graph traversal and relationship discovery happen, but the agents need that relationship structure materialized in Delta so they can compare it with unstructured documents in the same compute layer. Extraction is the bridge: it captures the rich relationships discovered in the graph and co-locates them with everything else in Unity Catalog. The lakehouse handles its own durability; extraction exists to feed the agents, not to back up Neo4j.

## Two Categories of Delta Tables

The pipeline produces two kinds of artifacts in Delta, and they have different lifecycles.

**Base tables** are the fourteen tables created from CSV files: seven node tables (customer, bank, account, company, stock, position, transaction) and seven relationship tables (has_account, at_bank, of_company, performs, benefits_to, has_position, of_security). These represent the original portfolio graph. They are stable, created once via `loading.py`, and re-created only if missing. They do not change between enrichment cycles.

**The enrichment log** is a single Delta table that accumulates every enrichment proposal across runs. Each row records the relationship type, source and target node references, confidence level, provenance metadata, and a JSON column for custom properties. Because the LLM can propose arbitrary new relationship types (INTERESTED_IN, CONCERNED_ABOUT, or types no one anticipated), storing them in one table with relationship type as a column avoids DDL for each new type. The data volumes are small: tens to low hundreds of rows per cycle.

## Why Dual-Write

When the pipeline writes a proposal to Neo4j, it simultaneously writes the same proposal to the enrichment log in Delta. This dual-write serves three purposes.

First, deduplication. Before proposing new relationships, the pipeline checks the enrichment log for existing entries matching the same source node, target node, and relationship type. Repeated cycles do not re-propose what has already been written.

Second, agent context. Prior enrichments are included in synthesis prompts so the LLM understands what relationships already exist in the graph. Without this context, the LLM rediscovers the same gaps cycle after cycle.

Third, incremental extraction signal. The enrichment log records what the pipeline itself has written. Extraction from Neo4j is needed for relationships created by external tools or manual edits, but the pipeline's own contributions are already tracked.

## Shared Schema

Today the schema is scattered across four files with incompatible representations. `loading.py` defines column types as hardcoded SQL CASTs. `seeding.py` maps Neo4j labels to Delta table names in a dictionary. `structured_data.py` maintains its own list of table names. `writeback.py` hardcodes five provenance properties. When the Neo4j Spark Connector infers types on extraction, it produces `LONG` where the CSV loader cast `INT`, or includes `<id>` and `<labels>` columns the base tables never had. The result is the `DELTA_FAILED_TO_MERGE_FIELDS` and `DELTA_METADATA_MISMATCH` errors the pipeline currently throws.

A shared schema registry (`graph_schema.py`) replaces all of these with a single source of truth. Each node type is defined once with its Neo4j label, Delta table name, key property, and column definitions including Delta types. Each relationship type is defined once with its source and target labels, keys, and the Delta table it derives from. Provenance properties (confidence, source_document, extracted_phrase, enrichment_timestamp, run_id) are defined once and referenced by the writeback layer.

The loading module generates its SQL CASTs from the schema. The seeding module derives its Neo4j label mappings from the schema. The extraction module casts Spark Connector output to match the schema before writing to Delta. The writeback module references provenance properties from the schema rather than hardcoding them. One definition, four consumers.

```python
@dataclass(frozen=True)
class PropertySchema:
    name: str
    delta_type: str       # STRING, INT, DOUBLE, DATE, BOOLEAN, TIMESTAMP
    required: bool = False
    csv_name: str | None = None  # when CSV column differs (holding_id → position_id)

@dataclass(frozen=True)
class NodeSchema:
    label: str            # Neo4j label: "Customer"
    table_name: str       # Delta table: "customer"
    key_property: str     # "customer_id"
    properties: tuple[PropertySchema, ...]
    csv_file: str | None = None

@dataclass(frozen=True)
class RelationshipSchema:
    rel_type: str         # Neo4j type: "HAS_ACCOUNT"
    table_name: str       # Delta table: "has_account"
    source_label: str
    source_key: str
    target_label: str
    target_key: str
    source_table: str     # Delta table to derive from
```

## Incremental Extraction

With the shared schema in place, extraction becomes both schema-safe and incremental.

For base tables, the pipeline checks whether they exist before extracting. If they are already populated from CSV loading, extraction skips them. These tables represent the original portfolio graph and do not change between cycles.

For enrichment relationships, the pipeline filters by `enrichment_timestamp` after the Spark Connector reads. The connector itself only supports label and relationship-type filtering, not property-level predicates, so the full relationship type is read and then filtered in Spark to rows newer than the last extraction. The data volumes are small enough that this is efficient.

For all writes, the schema registry provides the expected column types. The extraction module casts each DataFrame column to match the schema before writing, and uses `overwriteSchema` as a safety net. Type mismatches between the Spark Connector's inference and the declared schema are resolved before they reach Delta.

## The Enrichment Log Schema

| Column | Type |
|--------|------|
| run_id | STRING |
| enrichment_timestamp | TIMESTAMP |
| relationship_type | STRING |
| source_label | STRING |
| source_key_property | STRING |
| source_key_value | STRING |
| target_label | STRING |
| target_key_property | STRING |
| target_key_value | STRING |
| confidence | STRING |
| source_document | STRING |
| extracted_phrase | STRING |
| rationale | STRING |
| properties_json | STRING |

The `EnrichmentStore` class provides `ensure_table` (idempotent DDL), `write_proposals` (insert from InstanceProposal objects), `get_all` (full history for agent context), and `get_existing_keys` (bulk dedup lookup returning the set of source/target/relationship_type tuples already written).

## Pipeline Flow

The orchestrator (`run_graph_feature_forge.py`) changes from a linear extract-then-analyze pipeline to an incremental enrichment cycle.

```
 1. Authenticate to Databricks
 2. Ensure enrichment_log table exists
 3. Check if base tables exist; if not, load from CSV and seed Neo4j
 4. Extract incremental graph state (skip existing base tables, filter by timestamp)
 5. Load structured data from base tables
 6. Load prior enrichments from enrichment_log
 7. Include prior enrichments in synthesis context
 8. Configure DSPy and run analyzers
 9. Resolve schema suggestions to instance proposals
10. Deduplicate against enrichment_log
11. Filter by confidence
12. Dual-write HIGH proposals to Neo4j + enrichment_log
13. Save results JSON to UC Volume
```

Steps 6, 7, and 10 are new. Step 3 replaces the unconditional full extraction. The rest of the pipeline is unchanged.

## Files Changed

| File | Action | Change |
|------|--------|--------|
| `src/graph_feature_forge/graph_schema.py` | New | Shared schema registry: NodeSchema, RelationshipSchema, base definitions, utility functions |
| `src/graph_feature_forge/enrichment_store.py` | New | EnrichmentStore: Delta table for tracking proposals across runs |
| `src/graph_feature_forge/loading.py` | Modify | Generate SQL from graph_schema instead of hardcoded strings |
| `src/graph_feature_forge/seeding.py` | Modify | Import node/relationship mappings from graph_schema |
| `src/graph_feature_forge/structured_data.py` | Modify | Import table lists from graph_schema; add enrichment context method |
| `src/graph_feature_forge/extraction.py` | Modify | Schema-driven casting, overwriteSchema, incremental filtering, skip-if-exists |
| `src/graph_feature_forge/writeback.py` | Modify | Dual-write to Neo4j and enrichment_log |
| `agent_modules/run_graph_feature_forge.py` | Modify | Incremental pipeline flow with dedup and enrichment context |

## Tradeoffs

The dual-write between Neo4j and the enrichment log is not atomic. If the Neo4j write succeeds but the Delta write fails, the systems diverge. At the current data volumes (tens of rows per cycle) and with the idempotent MERGE pattern, the risk is low. The enrichment log can be reconciled against Neo4j by querying for relationships where `enrichment_timestamp IS NOT NULL`.

The single enrichment_log table trades schema precision for simplicity. Custom properties are stored as JSON rather than typed columns. For a pipeline producing tens of proposals per cycle, this is the right tradeoff. If enrichment volumes grow by orders of magnitude, promoting frequent relationship types to dedicated tables becomes worthwhile.

Incremental extraction filters after the Spark Connector reads the full relationship type. The connector cannot push predicates to Neo4j. For relationship types with millions of instances this would be wasteful, but the portfolio graph has hundreds of relationships per type, not millions.

## What Compounds

The first cycle captures the obvious: interest-holding gaps where a customer's documents mention sectors their portfolio does not cover. The second cycle starts from a richer graph. Prior INTERESTED_IN relationships are visible alongside the base HAS_POSITION edges, and the agents can identify clusters of customers with similar interest patterns that no one designed upfront. By the third cycle, agents cross-reference new market research documents against those visible interest communities, surfacing matches that span three layers of inference: customer preference, portfolio gap, and market opportunity. Each cycle's enrichments become the next cycle's context.
