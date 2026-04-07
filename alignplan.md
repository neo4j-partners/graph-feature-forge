# Alignment Plan: Minimum Viable Loop

The paper (graph_enrichment_v3.md) describes a seven-step loop. ALIGN.md identifies three gaps to close for the workshop to demonstrate one complete cycle. But there is a fourth gap that ALIGN.md does not call out: the pipeline never reads from Neo4j. This plan covers all four.

## Gap 0: Read from Neo4j (The Missing First Step)

**Current state.** The pipeline reads from 14 Delta tables in `neo4j_augmentation_demo.raw_data`. These were exported from Neo4j once during the workshop's Lab 4 using the Neo4j Spark Connector. The pipeline treats them as a static snapshot. There is no code that connects to Neo4j, runs Cypher, or triggers an extraction.

**Why this matters.** The paper's step 1 is "Extract: The Neo4j Spark Connector writes graph data to Delta Lake tables in the lakehouse." The loop depends on re-extraction: after write-back adds INTERESTED_IN relationships to Neo4j, the next cycle must re-extract so the Delta tables reflect the updated graph. Without this, even if write-back works, the second run analyzes the same stale data and proposes the same enrichments. There is no compounding.

**What to build.** A Neo4j extraction step using the Neo4j Spark Connector, matching the paper's architecture. The connector reads node labels and relationship types as Spark DataFrames and writes them directly to Delta tables. This is the same approach the workshop's Lab 4 uses, so the output format is already compatible with the existing structured data queries.

The extraction must dynamically discover what to pull. Each enrichment cycle may add new relationship types (INTERESTED_IN, CONCERNED_ABOUT, etc.) that did not exist in the original 14-table export. The extractor should query Neo4j's schema (via `db.labels()` and `db.relationshipTypes()`) to discover all current node labels and relationship types, then extract each one using the connector. This way the pipeline always operates on the full current graph state, including whatever previous runs wrote back.

The connector requires the Neo4j Spark Connector JAR on the classpath. On Databricks clusters, this is installed as a library. On serverless, it requires a custom environment spec that includes the JAR. The pipeline should fail with a clear error if the connector is not available rather than silently falling back.

The extraction is idempotent: overwrite the target tables so each run starts from the latest graph state.

The pipeline entry point gets a new flag: `--skip-extraction` to bypass this step during development when the Delta tables are already populated. This keeps the existing fast-iteration workflow intact.

Connection uses four environment variables: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`.

## Gap 1: Instance-Level Proposals

**Current state.** The four DSPy analyzers produce schema-level suggestions. NewEntitiesAnalyzer proposes "add a FINANCIAL_GOAL node type." ImpliedRelationshipsAnalyzer proposes "add an INTERESTED_IN relationship type between Customer and Sector." These are ontology proposals, not graph mutations. They say what kinds of things could exist, not which specific nodes and edges to create.

**What the paper needs.** A proposal that says: Customer C0001 (James Anderson) INTERESTED_IN Sector "Renewable Energy", confidence HIGH, extracted from document "Customer Profile - James Anderson", phrase "expressed strong interest in renewable energy investments." This names both endpoints, carries a confidence level, and traces back to evidence.

**What to build.** A fifth DSPy signature (resolution step) that runs after the current analyzers. It receives the schema-level suggestions, the synthesis text (which contains all the structured data context and document excerpts), and produces a list of concrete instance proposals. The LLM does the matching because accuracy matters more than cost. It knows which customer IDs appear in the evidence, which entity names are mentioned, and can reason about whether the evidence actually supports a specific instance.

The output model would be an `InstanceProposal` with fields: source node (label + key), target node (label + key), relationship type, properties, confidence (using the existing ConfidenceLevel enum), source document, extracted phrase, and a back-reference to the schema-level suggestion that spawned it.

The existing ConfidenceLevel enum (HIGH, MEDIUM, LOW) stays. The paper shows numeric scores like 0.92, but the enum is simpler, already works with DSPy, and avoids the calibration problem where the LLM assigns arbitrary floats that look precise but aren't. The three tiers map naturally to the paper's action buckets: HIGH auto-approves, MEDIUM approves with flag, LOW queues for review. If calibration against ground truth later shows the enum is too coarse, numeric scores can be added then.

## Gap 2: Confidence Filtering

**Current state.** The `compute_statistics` method counts high-confidence items but makes no decisions based on confidence. All suggestions are treated the same.

**What the paper needs.** Proposals sorted into action buckets where confidence level determines what happens next.

**What to build.** A filtering function that takes a list of instance proposals and partitions them by confidence level:

- HIGH: auto-approve, feed into write-back
- MEDIUM: approve with flag, logged for review
- LOW: queued for review, not written

This is a pure function over a list. The pipeline entry point calls it after the resolution step, reports counts per bucket, and passes the HIGH bucket to write-back.

## Gap 3: Write-Back to Neo4j

**Current state.** The pipeline ends at `print_response_summary()`. No Neo4j driver usage, no Cypher generation. The `neo4j` package is in the enrichment extras but unused.

**What the paper needs.** Approved instance proposals become MERGE statements executed against Neo4j. Each written relationship carries provenance properties: source document, extracted phrase, confidence level, enrichment timestamp, and a run identifier. MERGE is idempotent so re-running the same proposal is safe.

**What to build.** Three pieces:

1. A Cypher generator that takes an `InstanceProposal` and produces a MERGE statement. The pattern is always the same: MATCH source node by key, MATCH or MERGE target node by key, MERGE the relationship with provenance properties.

2. A Neo4j writer that opens a driver connection, runs the generated statements in a transaction, and reports what was written. Connection parameters (URI, auth) come from environment variables, same pattern as the existing Databricks config.

3. Integration into the pipeline entry point: after filtering, take the HIGH-confidence bucket, generate Cypher, execute, and log results. `--dry-run` is the default. Pass `--execute` to actually write to Neo4j, so the pipeline can be tested safely without a live instance.

The Neo4j connection variables (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`) are shared with Gap 0's extraction step.

## Pipeline After All Four Gaps Close

The complete pipeline becomes:

1. Extract graph state from Neo4j to Delta tables (Gap 0, skippable)
2. Load structured data from Delta tables (existing)
3. Retrieve relevant documents via embeddings (existing)
4. Synthesize gap analysis via LLM (existing)
5. Run four DSPy analyzers for schema-level suggestions (existing)
6. Resolve schema-level suggestions into instance proposals (Gap 1)
7. Filter instance proposals by confidence (Gap 2)
8. Write HIGH-confidence proposals back to Neo4j (Gap 3, dry-run by default)

A second run of this pipeline would extract a graph that now contains the relationships written in step 8, and the analyzers would see a different gap landscape.

## Execution Order

Build in order: Gap 0 (extraction), Gap 1 (instance resolution), Gap 2 (filtering), Gap 3 (write-back). Each produces a testable artifact:

- After Gap 0: the pipeline re-extracts and confirms the Delta tables match Neo4j's current state.
- After Gap 1: the pipeline prints instance proposals alongside the existing schema-level output.
- After Gap 2: the pipeline prints proposals sorted into three confidence buckets.
- After Gap 3: the pipeline prints Cypher in dry-run mode or writes to Neo4j with `--execute`.

## What This Plan Does Not Cover

ALIGN.md describes four long-term capabilities: ontology validation, graph algorithms, incremental processing, and feedback loops. This plan ignores all four. They matter for production but are not needed to demonstrate one complete loop.

This plan also does not address persisting results to Delta tables. That is a separate concern, though it would be natural to add alongside write-back.
