# Graph Enrichment Without Genie or Knowledge Agents

## The Problem With the Current Pipeline

The graph enrichment process in graph-augmented-ai-workshop follows a chain of manually provisioned Databricks AI/BI services. Lab 4 exports Neo4j nodes and relationships into 14 Delta Lake tables. Lab 5 wraps those tables in a Genie Space for structured queries and creates a Knowledge Agent that indexes 14 HTML documents from a Unity Catalog Volume. Lab 6 wires both into a Multi-Agent Supervisor endpoint. Lab 7 finally does the interesting work: it queries that MAS endpoint with gap analysis prompts, feeds the responses through four DSPy analyzers running in parallel, and produces structured enrichment proposals (new nodes, missing attributes, implied relationships, investment themes).

Three of those steps exist only to give the fourth step a way to ask questions across structured data and unstructured documents. The Genie Space, Knowledge Agent, and Multi-Agent Supervisor are intermediaries. They require manual UI configuration in the Databricks workspace, they produce opaque endpoint names that must be copy-pasted between labs, and they constrain the enrichment agent to single-turn Responses API calls through a custom DSPy adapter. None of the actual analytical reasoning happens inside those services. The reasoning happens in the DSPy modules and in whatever foundation model backs the MAS. The services just route queries to data.

The proposal: replace the Genie, Knowledge Agent, and MAS with direct LLM calls against Databricks model serving endpoints, reading structured data from Delta tables via Spark SQL and unstructured documents from Unity Catalog Volumes directly. Everything still runs on Databricks. The enrichment logic stays the same. The intermediary services disappear.

This work lives in the semantic-auth project rather than as a modification to graph-augmented-ai-workshop. The graph enrichment pipeline is the first phase of a larger project; the structured data access, document retrieval, and synthesis layers built here will be extended with additional capabilities in future phases. Keeping it separate avoids polluting the workshop repo with scope that goes beyond its pedagogical purpose.

## What the MAS Actually Does

Before designing the replacement, it helps to be precise about what the current MAS provides to Lab 7. The `MASClient` in `mas_client.py` sends five types of gap analysis queries: interest-holding gaps, risk profile alignment, data quality gaps, investment theme extraction, and a comprehensive query that combines all four. Each query is a long natural language prompt. The MAS routes it to the Genie agent (which translates natural language into SQL against the Delta tables) and the Knowledge Agent (which performs retrieval-augmented generation over the HTML documents). The MAS then synthesizes both responses into a single text answer.

That answer, a blob of unstructured text typically a few thousand characters, becomes the input to the DSPy analyzers. The analyzers use ChainOfThought signatures to extract structured Pydantic objects: `SuggestedNode`, `SuggestedRelationship`, `SuggestedAttribute`, and `InvestmentTheme`, each with confidence levels and source evidence.

So the MAS is performing two functions: data retrieval (querying Delta tables and searching documents) and synthesis (combining the results into a coherent gap analysis). Both of these can be done with direct LLM calls against data that's already accessible in the Databricks environment.

## The Replacement Architecture

The replacement has three layers, all running on Databricks compute clusters within the semantic-auth project.

### Layer 1: Structured Data Access via Spark SQL

The 14 Delta tables already exist in Unity Catalog from the Neo4j export. Instead of routing natural language through Genie to generate SQL, write the SQL directly. The gap analysis queries in the current system ask predictable, well-scoped questions: what stocks does customer C0001 hold, what is their risk profile, which sectors appear in their portfolio. These translate to a small number of Spark SQL queries against the node and relationship tables (customer, account, position, stock, transaction, and their join tables).

A structured data module would execute these queries against a Databricks cluster using the Databricks SDK's SQL execution capabilities or a Spark session, then format the results as context strings. The queries are deterministic and known in advance, so there's no need for natural language to SQL translation. If a future scenario requires dynamic queries, a foundation model endpoint can generate SQL from a prompt with the table schemas provided as context, but the current gap analysis questions don't require that flexibility.

### Layer 2: Document Retrieval via Embeddings and Vector Search

The 14 HTML documents currently live in a Unity Catalog Volume. The workshop already has pre-computed embeddings for these documents (20 chunks, 1024-dimensional vectors from `databricks-gte-large-en`). The Knowledge Agent is essentially performing RAG over these same documents.

The replacement would load the pre-computed embeddings (or regenerate them using the existing `generate_embeddings.py` script against the `databricks-gte-large-en` endpoint), store them in a Delta table with vector search indexing, and perform similarity search at query time. For each gap analysis question, embed the query, retrieve the top-k relevant document chunks, and include them as context in the LLM prompt.

Databricks provides vector search indexes on Delta tables through Mosaic AI Vector Search. The Databricks MCP server exposes this as three tools: `create_or_update_vs_endpoint` to provision compute, `create_or_update_vs_index` to create a DELTA_SYNC index with self-managed embeddings, and `query_vs_index` to run similarity queries. This means the entire retrieval pipeline can be set up and tested interactively via MCP before writing any module code. Alternatively, since the document corpus is small (20 chunks), a simple in-memory cosine similarity search would work without any additional infrastructure. The choice depends on whether this needs to scale beyond the workshop's 14 documents.

### Layer 3: LLM Synthesis via Foundation Model Endpoints

With structured query results and relevant document chunks in hand, the synthesis step sends a single prompt to a Databricks foundation model endpoint. The target model is Claude Sonnet 4.6 (`databricks-claude-sonnet-4-6`), available as a pay-per-token endpoint on Databricks. Any chat-completions-compatible endpoint works, but Claude Sonnet 4.6 provides strong reasoning, long-context handling, and reliable structured output extraction.

The synthesis prompt would include three sections: the structured data results (formatted customer portfolios, account balances, stock positions), the retrieved document chunks (customer profiles, market research excerpts), and the gap analysis instructions (adapted from the existing queries in `mas_client.py`). The model produces the same kind of gap analysis text that the MAS currently returns.

This replaces the custom `DatabricksResponsesLM` adapter in `config.py` with a standard `dspy.LM` pointing at an OpenAI-compatible chat endpoint. DSPy already supports this natively. The four analyzers (investment themes, new entities, missing attributes, implied relationships) continue to run in parallel via `dspy.Parallel` exactly as they do today, but against a standard model endpoint instead of the MAS.

## What Changes in the DSPy Layer

The DSPy modules, signatures, and schemas from Lab 7 don't change at all. The `GraphAugmentationAnalyzer`, its four sub-analyzers, and all the Pydantic output models stay the same. What changes is the input: instead of calling `fetch_gap_analysis()` which queries the MAS, the agent calls a new function that assembles the gap analysis context from the two direct data sources.

The DSPy LM configuration simplifies. The current `DatabricksResponsesLM` class exists solely because MAS endpoints use the Responses API format rather than standard chat completions. With a standard foundation model endpoint, the configuration becomes a single `dspy.LM` call with the endpoint URL and authentication token from `WorkspaceClient`. No custom adapter needed.

## What This Eliminates

Five manual setup steps disappear from the workshop:

1. Creating the Genie Space with table selections, instructions, SQL expressions, and sample questions (Lab 5, roughly 15 minutes of UI configuration).
2. Creating the Knowledge Agent with UC Volume source, content descriptions, and instructions (Lab 5, roughly 10 minutes).
3. Creating the Multi-Agent Supervisor with agent wiring and endpoint configuration (Lab 6, roughly 10 minutes).
4. Copying endpoint names between Labs 5, 6, and 7.
5. Writing and maintaining the `DatabricksResponsesLM` custom adapter for the non-standard Responses API format.

The replacement is entirely code-driven. The Databricks MCP server provides tools for every infrastructure interaction: `manage_uc_objects` for catalog/schema/volume creation, `execute_sql` for data queries, `upload_to_volume` for file management, `query_serving_endpoint` for model calls, and the vector search tools for retrieval. Infrastructure setup, data exploration, and prompt prototyping all happen through MCP tool calls. The validated logic then gets codified into Python modules. No UI clicks, no opaque endpoint names, no multi-service dependency chain.

## What This Costs

Removing the Genie layer means losing natural language to SQL translation. The current architecture lets the MAS answer ad-hoc questions about the structured data without anyone writing SQL. In the replacement, the SQL queries are pre-written for the known gap analysis scenarios. If someone wants to ask a new question about the data, they write a new query rather than typing a natural language question. For the graph enrichment use case this is fine since the questions are well-defined, but it's less flexible for open-ended exploration.

Removing the Knowledge Agent means building and maintaining the RAG pipeline directly: chunking, embedding, retrieval, and prompt assembly. The Knowledge Agent handles all of this as a managed service. The tradeoff is control versus convenience. With the direct approach, you can tune chunk sizes, overlap, retrieval strategies, and re-ranking in ways the Knowledge Agent doesn't expose. But you're also responsible for all of it.

The MAS provides agent orchestration, deciding which sub-agent to query based on the question. In the replacement, the orchestration is explicit in code: always query both data sources and include both in the synthesis prompt. This is simpler but less adaptable if the set of data sources grows.

## What Lives in semantic-auth

The semantic-auth project would contain:

- A structured data module that executes Spark SQL queries against the workshop's existing Delta tables and formats results as context strings.
- A document retrieval module that loads pre-computed embeddings and performs vector similarity search, preferably via Neo4j vector indexes, falling back to Mosaic AI Vector Search or in-memory cosine similarity.
- A synthesis module that assembles the structured data context, document chunks, and gap analysis instructions into a prompt, sends it to `databricks-claude-sonnet-4-6`, and returns the gap analysis text.
- An enrichment pipeline entry point that wires these three modules together and feeds the result into vendored copies of the DSPy analyzer modules from graph-augmented-ai-workshop, also backed by `databricks-claude-sonnet-4-6` with separate LM configuration.
- Configuration for Databricks workspace connection, model endpoint selection, and Delta table/Volume paths, targeting the `semantic-graph-enrichment` catalog.

The project runs on a Databricks cluster with access to Unity Catalog (for the Delta tables and Volumes) and the `databricks-claude-sonnet-4-6` and `databricks-gte-large-en` pay-per-token foundation model endpoints. No Genie, no Knowledge Agent, no Multi-Agent Supervisor.

## Sequencing the Build

The natural order is infrastructure first, then data access bottom-up, then the full pipeline. The Databricks MCP server provides tools that cover catalog management, SQL execution, volume operations, model serving queries, and vector search — enough to prototype and validate each phase interactively before writing any Python modules.

The development loop for each phase is: explore the data and test approaches via MCP tools, validate the outputs are correct, then codify the validated logic into Python modules. This means Phases 0–2 can be done almost entirely through interactive MCP tool calls.

Start with Phase 0: use `manage_uc_objects` to create the `semantic-graph-enrichment` catalog, schema, and volume. Use `upload_to_volume` to push the pre-computed embeddings. Verify cross-catalog access to `neo4j_augmentation_demo.raw_data` via `execute_sql`. Confirm both foundation model endpoints are reachable via `get_serving_endpoint_status` and `query_serving_endpoint`. Validate DSPy's `ChatAdapter` compatibility via `execute_code` on serverless compute — this is the compatibility gate for the entire project. This phase has no code dependencies and unblocks everything else.

Then the structured data module. Use `get_table_stats_and_schema` to discover all 14 Delta table schemas in `neo4j_augmentation_demo.raw_data`. Derive the gap analysis SQL queries from the five prompts in `mas_client.py`. Prototype and test them via `execute_sql` and `execute_sql_multi`, then codify the validated queries into the structured data module.

Next, the document retrieval module. Use `download_from_volume` to inspect the pre-computed embeddings locally. Evaluate whether Neo4j vector search can serve as the retrieval backend. If not, use the Mosaic AI Vector Search MCP tools (`create_or_update_vs_endpoint`, `create_or_update_vs_index`, `query_vs_index`) as the fallback — this path requires no custom retrieval code. Validate retrieval quality for each gap analysis query type via `query_vs_index`.

Then the synthesis module. Use `query_serving_endpoint` to prototype the synthesis prompt against `databricks-claude-sonnet-4-6` interactively. Combine both data sources and compare the output to what the MAS produces for the comprehensive gap analysis query. The output doesn't need to be identical, just informationally equivalent: it should surface the same interest-holding gaps, risk misalignments, and data quality issues.

Finally, wire the synthesis output into the DSPy analyzers (also backed by `databricks-claude-sonnet-4-6` with separate configuration). Use `execute_code` to run the full pipeline on serverless compute and verify the end-to-end output produces structured enrichment proposals comparable to the current Lab 7 output.

## Decisions

Answers to the key architectural and scoping questions, based on initial review.

### Data and Infrastructure

- The 14 Delta tables from the Neo4j export (Lab 4) already exist in `neo4j_augmentation_demo.raw_data`. semantic-auth does not re-run the export; it reads from that catalog via cross-catalog access.
- Start with the pre-computed embeddings JSON that ships with the workshop (`document_chunks_embedded.json`, 20 chunks, 1024 dimensions from `databricks-gte-large-en`). Re-chunking from raw HTML is a later optimization if retrieval quality needs tuning.
- For vector search, prefer Neo4j's native vector index since the graph is already in play and this keeps the retrieval path inside the same system that will receive the enrichment results. If Neo4j vector search proves impractical (version constraints, hosting limitations), fall back to Mosaic AI Vector Search on Databricks via the MCP tools (`create_or_update_vs_endpoint`, `create_or_update_vs_index`, `query_vs_index`). The Mosaic AI fallback path requires no custom retrieval code — the MCP tools handle endpoint provisioning, index creation with pre-computed embeddings, and similarity queries directly.

### Model Selection

- Use Claude Sonnet 4.6 via the Databricks pay-per-token foundation model endpoint: `databricks-claude-sonnet-4-6`. This is available as a pay-per-token endpoint in the Databricks workspace with no provisioned throughput commitment required.
- The synthesis step and the DSPy analyzers will use separate model configurations. Both can point at `databricks-claude-sonnet-4-6` but with different parameter settings: the synthesis step needs a higher max token limit to handle the combined structured data and document context, while the analyzers need tighter temperature settings for reliable structured output extraction. Separating them also leaves room to swap the analyzer model to a smaller or cheaper endpoint later without affecting synthesis quality.

### Scope and Integration

- Vendor the DSPy modules and Pydantic schemas from graph-augmented-ai-workshop into semantic-auth. The modules are ~500 lines total (schemas, signatures, analyzers). A cross-project import dependency is fragile and adds no value; vendoring gives semantic-auth freedom to diverge as the project evolves.
- The enrichment pipeline stops at producing the structured `AugmentationResponse` for now. Writing results back to Neo4j is a future phase.
- Gap analysis queries are hardcoded for the known scenarios. The queries are well-defined and the structured data module is purpose-built for them. User-defined enrichment questions are a future extension.
- Authentication follows the `WorkspaceClient` pattern from the workshop. It already handles both on-cluster (automatic runtime credentials) and local development (`DATABRICKS_HOST` + `DATABRICKS_TOKEN`) transparently. Service principal auth for automated runs is a future addition.

### Quality and Validation

- Validation uses a qualitative rubric rather than exact output comparison: does the gap analysis surface the known interest-holding gaps (James Anderson / renewable energy, Maria Rodriguez / ESG, Robert Chen / aggressive growth)? Do the DSPy analyzers produce the same categories of suggestions (new nodes, missing attributes, implied relationships, investment themes)?
- Start with manual spot-checking. A side-by-side comparison harness is useful but adds scope; build it only if manual review reveals ambiguous results.

## Phased Implementation

### Phase 0: Catalog and Workspace Setup

Provision the Databricks catalog, schema, and volume that semantic-auth will use. This follows the same pattern as the `databricks-neo4j-lab/lab_setup/catalog-validation` tooling, which creates a catalog, schema, volume, uploads data files, and grants permissions. The semantic-auth project needs its own namespace rather than writing into the workshop's `neo4j_augmentation_demo.raw_data`.

The recommended catalog name is `semantic-graph-enrichment`. This is descriptive of the project's purpose (semantic analysis for graph enrichment), avoids collision with the workshop catalog (`neo4j_augmentation_demo`), and follows the Databricks convention of lowercase-with-hyphens. The schema for enrichment artifacts would be `enrichment`, and the volume for document files and embeddings would be `source-data`.

Resulting paths:
- Catalog: `semantic-graph-enrichment`
- Schema: `semantic-graph-enrichment.enrichment`
- Volume: `/Volumes/semantic-graph-enrichment/enrichment/source-data/`

Nearly every step in this phase can be executed via Databricks MCP tools. On workspaces with Default Storage enabled, programmatic catalog creation requires a `MANAGED LOCATION` clause pointing to an existing storage root (e.g., the URL from an external location). The pattern is in `databricks-neo4j-lab/lab_setup/catalog-validation`. The schema, volume, and all downstream resources can be automated via MCP.

- [x] Verify workspace connectivity via `get_current_user`. Confirmed as `ryan.knight@neo4j.com` on `adb-1098933906466604.4.azuredatabricks.net`.
- [x] Create the catalog `semantic-graph-enrichment`. Default Storage blocks programmatic creation without a location — resolved by passing `MANAGED LOCATION 'abfss://unity-catalog-storage@dbstorageolrfaaff6x6nu.dfs.core.windows.net/1098933906466604'` (storage root from `partner_demo_workspace_v2` external location, pattern from `databricks-neo4j-lab/lab_setup/catalog-validation`). Created via `execute_code` on serverless using `spark.sql("CREATE CATALOG IF NOT EXISTS ... MANAGED LOCATION ...")`.
- [x] Create the schema `enrichment` within the catalog via `spark.sql("CREATE SCHEMA IF NOT EXISTS ...")` on serverless compute.
- [x] Create a managed volume `source-data` within the schema via `spark.sql("CREATE VOLUME IF NOT EXISTS ...")` on serverless compute.
- [x] Upload the pre-computed embeddings file via `upload_to_volume` (`document_chunks_embedded.json`, 557KB, 20 chunks, 1024 dimensions). Verified at `/Volumes/semantic-graph-enrichment/enrichment/source-data/embeddings/document_chunks_embedded.json`.
- [x] Verify that the 14 Delta tables from the workshop's Neo4j export are accessible via `execute_sql` (`SHOW TABLES IN neo4j_augmentation_demo.raw_data`). All 14 tables confirmed: account, at_bank, bank, benefits_to, company, customer, has_account, has_position, of_company, of_security, performs, position, stock, transaction.
- [x] Grant Unity Catalog permissions via `manage_uc_grants`: `USE_CATALOG`, `USE_SCHEMA`, `SELECT`, `READ_VOLUME` granted to `account users` on the `semantic-graph-enrichment` catalog.
- [x] Verify that the `databricks-claude-sonnet-4-6` endpoint is accessible. Status: READY. Tested with chat completion — responds correctly.
- [x] Verify that DSPy's `ChatAdapter` works correctly with `databricks-claude-sonnet-4-6` via `execute_code` on serverless compute. DSPy 3.0.4 `ChainOfThought` produced correct answer with reasoning. **Key constraint**: serverless runtime (Python 3.10) requires `pydantic>=2.10,<2.12` due to `typing_extensions` lacking `Sentinel`. Pin order: `typing_extensions>=4.12` first, then `pydantic`, then `dspy`.
- [x] Verify that the `databricks-gte-large-en` embedding endpoint is accessible. Status: READY. Tested via `execute_code` — returns 1024-dimensional vectors.
- [x] Set up the semantic-auth project structure: `pyproject.toml` (with pydantic<2.12 constraint), `.env.example`, `config.py`, `structured_data.py` already in place from prior work.

### Phase 1: Structured Data Access

Stand up the Spark SQL layer that replaces what Genie does. The goal is a module that takes a gap analysis query type (interest-holding gaps, risk alignment, data quality, investment themes) and returns formatted structured data context as a string, ready to be included in an LLM prompt. The 14 Delta tables already exist in `neo4j_augmentation_demo.raw_data` from the workshop's Neo4j export; this phase reads from them, it does not create them.

The MCP tools enable an explore-first workflow for this phase: use `get_table_stats_and_schema` to discover all table schemas in a single call, then use `execute_sql` to iteratively develop and test each gap analysis query before codifying them into the Python module.

- [x] Identify the exact table names for the 14 Delta tables in `neo4j_augmentation_demo.raw_data`. Use `get_table_stats_and_schema` (`catalog: "neo4j_augmentation_demo"`, `schema: "raw_data"`) to discover all tables, their column names, types, and row counts in one call. The 7 node tables are: customer, bank, account, company, stock, position, transaction. The 7 relationship tables are: has_account (Customer→Account), at_bank (Account→Bank), of_company (Stock→Company), performs (Account→Transaction), benefits_to (Transaction→Account), has_position (Account→Position), of_security (Position→Stock). Note: the export script defaults to schema `graph_data` — verify which schema is actually populated on the workspace.
- [x] Review the five gap analysis prompts in `mas_client.py` and translate each into SQL queries. Prototype each query via `execute_sql` to validate results before codifying. Use `execute_sql_multi` to run all gap analysis queries in parallel once finalized. Three queries implemented: (1) portfolio holdings — 8-way join from customer through has_account, account, has_position, position, of_security, stock, of_company, company; grouped by customer and sector. (2) customer profiles — customer demographics + account summary via has_account join. (3) data completeness — all customer fields with null analysis and schema gap identification. Investment themes targets documents only, no SQL needed.
- [x] Build a structured data module (`src/semantic_auth/structured_data.py`) with a pluggable SQL executor pattern: `make_spark_executor()` for on-cluster use, `make_sdk_executor(warehouse_id)` for local/remote use. `StructuredDataAccess` class provides `get_portfolio_holdings()`, `get_customer_profiles()`, `get_data_completeness()`, and `get_all_structured_context()`. Includes `discover_schema()` to verify actual column names before running queries. Note: the customer table has no risk_profile field — risk tolerance information exists only in the unstructured HTML documents.
- [ ] Test the module against the Delta tables. Use `execute_sql` to verify the actual relationship table column format (`source.X` / `target.X` prefixes from the Neo4j Spark Connector with `relationship.nodes.map=false`). Run schema discovery via `get_table_stats_and_schema` with `table_stat_level: "DETAILED"` on the relationship tables to confirm column names, then validate that the structured context for each query type contains the expected data.
- [x] Document the SQL queries and their mapping to the original gap analysis prompt types. Query-to-prompt mapping documented in structured_data.py method docstrings. Gap analysis type → SQL mapping: interest-holding gaps → `get_portfolio_holdings()`, risk alignment → `get_customer_profiles()`, data quality → `get_data_completeness()`, investment themes → documents only (Phase 2), comprehensive → `get_all_structured_context()`.

### Phase 2: Document Retrieval

Build the RAG pipeline that replaces what the Knowledge Agent does. The goal is a module that takes a query string, retrieves the most relevant document chunks, and returns them as context strings. Start with the pre-computed embeddings JSON; evaluate Neo4j vector search as the primary retrieval backend.

If Neo4j vector search is not available, the Mosaic AI Vector Search MCP tools provide a complete fallback path that requires no custom retrieval code: `create_or_update_vs_endpoint` provisions compute, `create_or_update_vs_index` creates a DELTA_SYNC index over a Delta table of pre-computed embeddings, and `query_vs_index` runs similarity queries. This path can be set up and validated entirely through MCP tool calls before deciding whether to codify it into a module.

- [ ] Load the pre-computed embeddings from the volume. Use `download_from_volume` (`volume_path: "/Volumes/semantic-graph-enrichment/enrichment/source-data/embeddings/document_chunks_embedded.json"`) to pull the file locally and inspect chunk structure, embedding dimensions, and metadata. This file contains 20 chunks with 1024-dimensional vectors from `databricks-gte-large-en`.
- [ ] Evaluate Neo4j vector search as the retrieval backend. If the Neo4j instance supports vector indexes (Neo4j 5.11+ or Aura), create a vector index on a `Chunk` node type and load the pre-computed embeddings as chunk nodes with their vectors. This keeps the retrieval path inside the same database that receives enrichment results, and enables hybrid queries that combine vector similarity with graph traversal (e.g., "find chunks related to customers who hold tech stocks").
- [ ] If Neo4j vector search is not available (version constraints or hosting limitations), set up Mosaic AI Vector Search via MCP tools. First, load the embeddings JSON into a Delta table via `execute_sql` (CREATE TABLE with the chunk text, metadata, and embedding vector columns). Then provision a vector search endpoint via `create_or_update_vs_endpoint` (`name: "semantic-auth-vs"`, `endpoint_type: "STANDARD"`). Create a DELTA_SYNC index with self-managed embeddings via `create_or_update_vs_index` (`endpoint_name: "semantic-auth-vs"`, `primary_key: "chunk_id"`, `delta_sync_index_spec` with `embedding_vector_columns: [{"name": "embedding", "embedding_dimension": 1024}]`). As a lighter alternative, build an in-memory retrieval module that loads the chunk embeddings, embeds a query string using `query_serving_endpoint` against `databricks-gte-large-en`, computes cosine similarity, and returns the top-k chunks (start with k=5).
- [ ] Test retrieval quality for each gap analysis query type. If using Mosaic AI Vector Search, use `query_vs_index` with `query_vector` (pre-computed via `query_serving_endpoint` against `databricks-gte-large-en`) to validate results interactively. For interest-holding gaps, the module should retrieve customer profile chunks mentioning investment interests (James Anderson's renewable energy interest, Maria Rodriguez's ESG interest, Robert Chen's technology interest). For investment themes, it should retrieve market research and sector analysis chunks.
- [ ] If retrieval quality is insufficient with the pre-computed chunks, experiment with re-chunking the raw HTML documents at different sizes or adding metadata-based filtering (retrieve only customer profiles for customer-specific queries, only market research for theme queries).

### Phase 3: LLM Synthesis

Replace the MAS with a single synthesis step that combines structured data and document chunks into a gap analysis prompt and sends it to `databricks-claude-sonnet-4-6`. Use `query_serving_endpoint` to prototype the prompt template interactively before codifying it into a module.

- [ ] Design the prompt template first by iterating via `query_serving_endpoint` (`name: "databricks-claude-sonnet-4-6"`). Assemble sample structured data context (from Phase 1's `execute_sql` results) and sample document chunks (from Phase 2's retrieval output) into a messages payload. The prompt needs three clearly delineated sections: structured data (labeled so the model knows this is factual portfolio/account data), document excerpts (labeled as qualitative profile and research content), and analysis instructions (adapted from the existing gap analysis queries). The instructions should explicitly ask the model to compare structured holdings against document-stated interests and identify gaps. Iterate on the prompt directly via `query_serving_endpoint` until the output quality is satisfactory.
- [ ] Build a synthesis module (`synthesis.py` or similar) that codifies the validated prompt template. The module takes structured data context (from Phase 1) and document chunks (from Phase 2), assembles them into the tested prompt, and sends it to the `databricks-claude-sonnet-4-6` endpoint via the Databricks SDK (not MCP — the MCP tools are for prototyping, the module uses the SDK for production calls).
- [ ] Test the synthesis output against the known gap analysis results. At minimum, the output should identify: James Anderson holds tech stocks but expresses renewable energy interest; Maria Rodriguez has ESG/sustainable investing preferences not reflected in holdings; Robert Chen's aggressive growth appetite may not match his current portfolio composition.
- [ ] Compare token usage and latency against the MAS-backed approach. The MAS involves multiple round-trips (MAS to Genie, MAS to Knowledge Agent, MAS synthesis). The direct approach is a single LLM call with a larger context window. Measure whether the single-call approach is faster and more cost-effective.

### Phase 4: DSPy Integration

Wire the synthesis output into the existing DSPy analyzers, replacing the `DatabricksResponsesLM` and MAS dependency with `databricks-claude-sonnet-4-6` as a standard chat-completions endpoint. The synthesis step and the analyzers use separate DSPy LM configurations: the synthesis call uses a higher max token limit for context assembly, while the analyzers use tighter temperature for structured output extraction. Use `execute_code` on serverless compute to run and validate the full pipeline without requiring a local Databricks Connect setup.

- [ ] Configure DSPy with a standard `dspy.LM` pointing at `databricks-claude-sonnet-4-6` for the analyzer calls. The ChatAdapter compatibility with this endpoint was validated in Phase 0 via `execute_code`. Use a separate LM configuration for the synthesis step (higher max_tokens) versus the analyzer step (lower temperature, tighter output constraints).
- [ ] Vendor the DSPy modules from graph-augmented-ai-workshop into semantic-auth: `analyzers.py` (the four analyzer modules and `GraphAugmentationAnalyzer`), `signatures.py` (the DSPy signatures), and `schemas.py` (the Pydantic output models). These are ~500 lines total and will diverge as the project evolves.
- [ ] Build the pipeline entry point that calls the synthesis module from Phase 3, feeds the output into `GraphAugmentationAnalyzer`, and returns an `AugmentationResponse`. This replaces `agent_dspy.py`'s `main()` function.
- [ ] Run the full pipeline end-to-end via `execute_code` on serverless compute. Upload the pipeline code via `execute_code` with `workspace_path` to persist it as a notebook, then execute. Verify that the four parallel analyzers produce structured output: `SuggestedNode` objects (e.g., FINANCIAL_GOAL, STATED_INTEREST), `SuggestedRelationship` objects (e.g., INTERESTED_IN, HAS_GOAL), `SuggestedAttribute` objects (e.g., Customer.occupation, Customer.investment_philosophy), and `InvestmentTheme` objects.
- [ ] Enable MLflow tracing (`mlflow.dspy.autolog()`) and verify that traces capture the full pipeline: synthesis call, four parallel analyzer calls, and structured output extraction.
- [ ] Spot-check the `AugmentationResponse` output against the known gap analysis results. Verify that the suggestions cover the same categories and key findings as the MAS-backed pipeline. Document any significant differences in suggestion quality, confidence levels, or coverage.

### Phase 5: Packaging and Documentation

Make the project runnable as a standalone Databricks workflow.

- [ ] Create a Databricks notebook that runs the full pipeline via `execute_code` with `workspace_path` to persist it at a known workspace location. The notebook should be self-contained, requiring only a cluster with Unity Catalog access and foundation model endpoint access. Alternatively, use `execute_code` with `compute_type: "serverless"` if the pipeline can run without a persistent cluster.
- [ ] Write configuration documentation: which environment variables to set, which catalog/schema to target, which model endpoint to use, and how to switch between local development (`.env` file), on-cluster execution (automatic `WorkspaceClient` auth), and MCP-based interactive development (using the Databricks MCP tools for exploration and prototyping).
- [ ] Add a `pyproject.toml` with the necessary dependencies: `dspy`, `mlflow`, `databricks-sdk`, `databricks-langchain` (for embeddings), `pydantic`, `python-dotenv`, and `neo4j` (if using Neo4j vector search).
- [ ] Write a brief README for semantic-auth explaining what the project does, how it relates to graph-augmented-ai-workshop, and how to run it.
