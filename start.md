# Graph Enrichment Without Genie or Knowledge Agents

## The Problem With the Current Pipeline

The graph enrichment process in graph-augmented-ai-workshop follows a chain of manually provisioned Databricks AI/BI services. Lab 4 exports Neo4j nodes and relationships into 14 Delta Lake tables. Lab 5 wraps those tables in a Genie Space for structured queries and creates a Knowledge Agent that indexes 14 HTML documents from a Unity Catalog Volume. Lab 6 wires both into a Supervisor Agent endpoint. Lab 7 finally does the interesting work: it queries that Supervisor Agent endpoint with gap analysis prompts, feeds the responses through four DSPy analyzers running in parallel, and produces structured enrichment proposals (new nodes, missing attributes, implied relationships, investment themes).

Three of those steps exist only to give the fourth step a way to ask questions across structured data and unstructured documents. The Genie Space, Knowledge Agent, and Multi-Agent Supervisor are intermediaries. They require manual UI configuration in the Databricks workspace, they produce opaque endpoint names that must be copy-pasted between labs, and they constrain the enrichment agent to single-turn Responses API calls through a custom DSPy adapter. None of the actual analytical reasoning happens inside those services. The reasoning happens in the DSPy modules and in whatever foundation model backs the Supervisor Agent. The services just route queries to data.

The proposal: replace the Genie, Knowledge Agent, and Supervisor Agent with direct LLM calls against Databricks model serving endpoints, reading structured data from Delta tables via Spark SQL and unstructured documents from Unity Catalog Volumes directly. Everything still runs on Databricks. The enrichment logic stays the same. The intermediary services disappear.

This work lives in the semantic-auth project rather than as a modification to graph-augmented-ai-workshop. The graph enrichment pipeline is the first phase of a larger project; the structured data access, document retrieval, and synthesis layers built here will be extended with additional capabilities in future phases. Keeping it separate avoids polluting the workshop repo with scope that goes beyond its pedagogical purpose.

## What the Supervisor Agent Actually Does

Before designing the replacement, it helps to be precise about what the current Supervisor Agent provides to Lab 7. The `MASClient` in `mas_client.py` sends five types of gap analysis queries: interest-holding gaps, risk profile alignment, data quality gaps, investment theme extraction, and a comprehensive query that combines all four. Each query is a long natural language prompt. The Supervisor Agent routes it to the Genie agent (which translates natural language into SQL against the Delta tables) and the Knowledge Agent (which performs retrieval-augmented generation over the HTML documents). The Supervisor Agent then synthesizes both responses into a single text answer.

That answer, a blob of unstructured text typically a few thousand characters, becomes the input to the DSPy analyzers. The analyzers use ChainOfThought signatures to extract structured Pydantic objects: `SuggestedNode`, `SuggestedRelationship`, `SuggestedAttribute`, and `InvestmentTheme`, each with confidence levels and source evidence.

So the Supervisor Agent is performing two functions: data retrieval (querying Delta tables and searching documents) and synthesis (combining the results into a coherent gap analysis). Both of these can be done with direct LLM calls against data that's already accessible in the Databricks environment.

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

The synthesis prompt would include three sections: the structured data results (formatted customer portfolios, account balances, stock positions), the retrieved document chunks (customer profiles, market research excerpts), and the gap analysis instructions (adapted from the existing queries in `mas_client.py`). The model produces the same kind of gap analysis text that the Supervisor Agent currently returns.

This replaces the custom `DatabricksResponsesLM` adapter in `config.py` with a standard `dspy.LM` pointing at an OpenAI-compatible chat endpoint. DSPy already supports this natively. The four analyzers (investment themes, new entities, missing attributes, implied relationships) continue to run in parallel via `dspy.Parallel` exactly as they do today, but against a standard model endpoint instead of the Supervisor Agent.

## What Changes in the DSPy Layer

The DSPy modules, signatures, and schemas from Lab 7 don't change at all. The `GraphAugmentationAnalyzer`, its four sub-analyzers, and all the Pydantic output models stay the same. What changes is the input: instead of calling `fetch_gap_analysis()` which queries the Supervisor Agent, the agent calls a new function that assembles the gap analysis context from the two direct data sources.

The DSPy LM configuration simplifies. The current `DatabricksResponsesLM` class exists solely because Supervisor Agent endpoints use the Responses API format rather than standard chat completions. With a standard foundation model endpoint, the configuration becomes a single `dspy.LM` call with the endpoint URL and authentication token from `WorkspaceClient`. No custom adapter needed.

## What This Eliminates

Five manual setup steps disappear from the workshop:

1. Creating the Genie Space with table selections, instructions, SQL expressions, and sample questions (Lab 5, roughly 15 minutes of UI configuration).
2. Creating the Knowledge Agent with UC Volume source, content descriptions, and instructions (Lab 5, roughly 10 minutes).
3. Creating the Multi-Agent Supervisor with agent wiring and endpoint configuration (Lab 6, roughly 10 minutes).
4. Copying endpoint names between Labs 5, 6, and 7.
5. Writing and maintaining the `DatabricksResponsesLM` custom adapter for the non-standard Responses API format.

The replacement is entirely code-driven. The Databricks MCP server provides tools for every infrastructure interaction: `manage_uc_objects` for catalog/schema/volume creation, `execute_sql` for data queries, `upload_to_volume` for file management, `query_serving_endpoint` for model calls, and the vector search tools for retrieval. Infrastructure setup, data exploration, and prompt prototyping all happen through MCP tool calls. The validated logic then gets codified into Python modules. No UI clicks, no opaque endpoint names, no multi-service dependency chain.

## What This Costs

Removing the Genie layer means losing natural language to SQL translation. The current architecture lets the Supervisor Agent answer ad-hoc questions about the structured data without anyone writing SQL. In the replacement, the SQL queries are pre-written for the known gap analysis scenarios. If someone wants to ask a new question about the data, they write a new query rather than typing a natural language question. For the graph enrichment use case this is fine since the questions are well-defined, but it's less flexible for open-ended exploration.

Removing the Knowledge Agent means building and maintaining the RAG pipeline directly: chunking, embedding, retrieval, and prompt assembly. The Knowledge Agent handles all of this as a managed service. The tradeoff is control versus convenience. With the direct approach, you can tune chunk sizes, overlap, retrieval strategies, and re-ranking in ways the Knowledge Agent doesn't expose. But you're also responsible for all of it.

The Supervisor Agent provides agent orchestration, deciding which sub-agent to query based on the question. In the replacement, the orchestration is explicit in code: always query both data sources and include both in the synthesis prompt. This is simpler but less adaptable if the set of data sources grows.

## What Lives in semantic-auth

The semantic-auth project would contain:

- A structured data module that executes Spark SQL queries against the workshop's existing Delta tables and formats results as context strings.
- A document retrieval module that loads pre-computed embeddings and performs vector similarity search, preferably via Neo4j vector indexes, falling back to Mosaic AI Vector Search or in-memory cosine similarity.
- A synthesis module that assembles the structured data context, document chunks, and gap analysis instructions into a prompt, sends it to `databricks-claude-sonnet-4-6`, and returns the gap analysis text.
- An enrichment pipeline entry point that wires these three modules together and feeds the result into vendored copies of the DSPy analyzer modules from graph-augmented-ai-workshop, also backed by `databricks-claude-sonnet-4-6` with separate LM configuration.
- Configuration for Databricks workspace connection, model endpoint selection, and Delta table/Volume paths, targeting the `semantic-auth` catalog.

The project runs on a Databricks cluster with access to Unity Catalog (for the Delta tables and Volumes) and the `databricks-claude-sonnet-4-6` and `databricks-gte-large-en` pay-per-token foundation model endpoints. No Genie, no Knowledge Agent, no Multi-Agent Supervisor.

## Sequencing the Build

The natural order is infrastructure first, then data access bottom-up, then the full pipeline. The Databricks MCP server provides tools that cover catalog management, SQL execution, volume operations, model serving queries, and vector search — enough to prototype and validate each phase interactively before writing any Python modules.

The development loop for each phase is: explore the data and test approaches via MCP tools, validate the outputs are correct, then codify the validated logic into Python modules. This means Phases 0–2 can be done almost entirely through interactive MCP tool calls.

Start with Phase 0: use `manage_uc_objects` to create the `semantic-auth` catalog, schema, and volume. Use `upload_to_volume` to push the pre-computed embeddings. Verify cross-catalog access to `neo4j_augmentation_demo.raw_data` via `execute_sql`. Confirm both foundation model endpoints are reachable via `get_serving_endpoint_status` and `query_serving_endpoint`. Validate DSPy's `ChatAdapter` compatibility via `execute_code` on serverless compute — this is the compatibility gate for the entire project. This phase has no code dependencies and unblocks everything else.

Then the structured data module. Use `get_table_stats_and_schema` to discover all 14 Delta table schemas in `neo4j_augmentation_demo.raw_data`. Derive the gap analysis SQL queries from the five prompts in `mas_client.py`. Prototype and test them via `execute_sql` and `execute_sql_multi`, then codify the validated queries into the structured data module.

Next, the document retrieval module. Use `download_from_volume` to inspect the pre-computed embeddings locally. Evaluate whether Neo4j vector search can serve as the retrieval backend. If not, use the Mosaic AI Vector Search MCP tools (`create_or_update_vs_endpoint`, `create_or_update_vs_index`, `query_vs_index`) as the fallback — this path requires no custom retrieval code. Validate retrieval quality for each gap analysis query type via `query_vs_index`.

Then the synthesis module. Use `query_serving_endpoint` to prototype the synthesis prompt against `databricks-claude-sonnet-4-6` interactively. Combine both data sources and compare the output to what the Supervisor Agent produces for the comprehensive gap analysis query. The output doesn't need to be identical, just informationally equivalent: it should surface the same interest-holding gaps, risk misalignments, and data quality issues.

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

The recommended catalog name is `semantic-auth`. This is descriptive of the project's purpose (semantic analysis for graph enrichment), avoids collision with the workshop catalog (`neo4j_augmentation_demo`), and follows the Databricks convention of lowercase-with-hyphens. The schema for enrichment artifacts would be `enrichment`, and the volume for document files and embeddings would be `source-data`.

Resulting paths:
- Catalog: `semantic-auth`
- Schema: `semantic-auth.enrichment`
- Volume: `/Volumes/semantic-auth/enrichment/source-data/`

Nearly every step in this phase can be executed via Databricks MCP tools. On workspaces with Default Storage enabled, programmatic catalog creation requires a `MANAGED LOCATION` clause pointing to an existing storage root (e.g., the URL from an external location). The pattern is in `databricks-neo4j-lab/lab_setup/catalog-validation`. The schema, volume, and all downstream resources can be automated via MCP.

- [x] Verify workspace connectivity via `get_current_user`. Confirmed as `ryan.knight@neo4j.com` on `adb-1098933906466604.4.azuredatabricks.net`.
- [x] Create the catalog `semantic-auth`. Default Storage blocks programmatic creation without a location — resolved by passing `MANAGED LOCATION 'abfss://unity-catalog-storage@dbstorageolrfaaff6x6nu.dfs.core.windows.net/1098933906466604'` (storage root from `partner_demo_workspace_v2` external location, pattern from `databricks-neo4j-lab/lab_setup/catalog-validation`). Created via `execute_code` on serverless using `spark.sql("CREATE CATALOG IF NOT EXISTS ... MANAGED LOCATION ...")`.
- [x] Create the schema `enrichment` within the catalog via `spark.sql("CREATE SCHEMA IF NOT EXISTS ...")` on serverless compute.
- [x] Create a managed volume `source-data` within the schema via `spark.sql("CREATE VOLUME IF NOT EXISTS ...")` on serverless compute.
- [x] Upload the pre-computed embeddings file via `upload_to_volume` (`document_chunks_embedded.json`, 557KB, 20 chunks, 1024 dimensions). Verified at `/Volumes/semantic-auth/enrichment/source-data/embeddings/document_chunks_embedded.json`.
- [x] Verify that the 14 Delta tables from the workshop's Neo4j export are accessible via `execute_sql` (`SHOW TABLES IN neo4j_augmentation_demo.raw_data`). All 14 tables confirmed: account, at_bank, bank, benefits_to, company, customer, has_account, has_position, of_company, of_security, performs, position, stock, transaction.
- [x] Grant Unity Catalog permissions via `manage_uc_grants`: `USE_CATALOG`, `USE_SCHEMA`, `SELECT`, `READ_VOLUME` granted to `account users` on the `semantic-auth` catalog.
- [x] Verify that the `databricks-claude-sonnet-4-6` endpoint is accessible. Status: READY. Tested with chat completion — responds correctly.
- [x] Verify that DSPy's `ChatAdapter` works correctly with `databricks-claude-sonnet-4-6` via `execute_code` on serverless compute. DSPy 3.0.4 `ChainOfThought` produced correct answer with reasoning. **Key constraint**: serverless runtime (Python 3.10) requires `pydantic>=2.10,<2.12` due to `typing_extensions` lacking `Sentinel`. Pin order: `typing_extensions>=4.12` first, then `pydantic`, then `dspy`.
- [x] Verify that the `databricks-gte-large-en` embedding endpoint is accessible. Status: READY. Tested via `execute_code` — returns 1024-dimensional vectors.
- [x] Set up the semantic-auth project structure: `pyproject.toml` (with pydantic<2.12 constraint), `.env.example`, `config.py`, `structured_data.py` already in place from prior work.

### Phase 1: Structured Data Access

Stand up the Spark SQL layer that replaces what Genie does. The goal is a module that takes a gap analysis query type (interest-holding gaps, risk alignment, data quality, investment themes) and returns formatted structured data context as a string, ready to be included in an LLM prompt. The 14 Delta tables already exist in `neo4j_augmentation_demo.raw_data` from the workshop's Neo4j export; this phase reads from them, it does not create them.

The MCP tools enable an explore-first workflow for this phase: use `get_table_stats_and_schema` to discover all table schemas in a single call, then use `execute_sql` to iteratively develop and test each gap analysis query before codifying them into the Python module.

- [x] Identify the exact table names for the 14 Delta tables in `neo4j_augmentation_demo.raw_data`. Use `get_table_stats_and_schema` (`catalog: "neo4j_augmentation_demo"`, `schema: "raw_data"`) to discover all tables, their column names, types, and row counts in one call. The 7 node tables are: customer, bank, account, company, stock, position, transaction. The 7 relationship tables are: has_account (Customer→Account), at_bank (Account→Bank), of_company (Stock→Company), performs (Account→Transaction), benefits_to (Transaction→Account), has_position (Account→Position), of_security (Position→Stock). Note: the export script defaults to schema `graph_data` — verify which schema is actually populated on the workspace.
- [x] Review the five gap analysis prompts in `mas_client.py` and translate each into SQL queries. Prototype each query via `execute_sql` to validate results before codifying. Use `execute_sql_multi` to run all gap analysis queries in parallel once finalized. Three queries implemented: (1) portfolio holdings — 8-way join from customer through has_account, account, has_position, position, of_security, stock, of_company, company; grouped by customer and sector. (2) customer profiles — customer demographics + account summary via has_account join. (3) data completeness — all customer fields with null analysis and schema gap identification. Investment themes targets documents only, no SQL needed.
- [x] Build a structured data module (`src/semantic_auth/structured_data.py`) with a pluggable SQL executor pattern: `make_spark_executor()` for on-cluster use, `make_sdk_executor(warehouse_id)` for local/remote use. `StructuredDataAccess` class provides `get_portfolio_holdings()`, `get_customer_profiles()`, `get_data_completeness()`, and `get_all_structured_context()`. Includes `discover_schema()` to verify actual column names before running queries. The customer table includes `risk_profile` (Aggressive/Conservative/Moderate, 100% populated) and `employment_status`.
- [x] Test the module against the Delta tables. Verified `source.X` / `target.X` column format on all 7 relationship tables via `execute_code` on serverless. Portfolio holdings 8-way join returns 110 rows across 37 customers and 11 sectors. Customer profiles query returns all 102 customers with `risk_profile` and `employment_status`. Fixed: added `risk_profile`/`employment_status` to `get_customer_profiles()` SELECT and formatting, removed `risk_profile` from the "NOT IN SCHEMA" list in `_format_data_completeness()`, corrected docstrings.
- [x] Document the SQL queries and their mapping to the original gap analysis prompt types. Query-to-prompt mapping documented in structured_data.py method docstrings. Gap analysis type → SQL mapping: interest-holding gaps → `get_portfolio_holdings()`, risk alignment → `get_customer_profiles()`, data quality → `get_data_completeness()`, investment themes → documents only (Phase 2), comprehensive → `get_all_structured_context()`.

### Phase 2: Document Retrieval

Build the RAG pipeline that replaces what the Knowledge Agent does. The goal is a module that takes a query string, retrieves the most relevant document chunks, and returns them as context strings. Start with the pre-computed embeddings JSON; evaluate Neo4j vector search as the primary retrieval backend.

If Neo4j vector search is not available, the Mosaic AI Vector Search MCP tools provide a complete fallback path that requires no custom retrieval code: `create_or_update_vs_endpoint` provisions compute, `create_or_update_vs_index` creates a DELTA_SYNC index over a Delta table of pre-computed embeddings, and `query_vs_index` runs similarity queries. This path can be set up and validated entirely through MCP tool calls before deciding whether to codify it into a module.

- [x] Load the pre-computed embeddings from the volume. Inspected via `execute_code` on serverless. File structure: top-level dict with `metadata`, `documents` (14 entries), and `chunks` (20 entries). Each chunk has `chunk_id` (uuid), `index`, `text`, `document_id`, `metadata` (with `document_title`, `document_type`, `source_path`), and `embedding` (1024-dim float array). Chunk text lengths range from 77 to 4094 chars. Document types include `customer_profile`, `bank_branch`, `bank_profile`, `market_analysis`, `investment_guide`, `regulatory`.
- [x] Evaluate Neo4j vector search as the retrieval backend. No Neo4j connection is configured in the project (no NEO4J_URI in .env or .env.example). With only 20 chunks, in-memory cosine similarity is the pragmatic choice — zero infrastructure, sub-millisecond search. Neo4j vector search can be added in a future phase when the enrichment write-back needs it.
- [x] Built in-memory retrieval module (`src/semantic_auth/retrieval.py`). `DocumentRetrieval` class loads chunks from JSON (via `from_json_path` or `from_json_str`), embeds queries via a pluggable `Embedder` callable, computes cosine similarity, and returns top-k `RetrievedChunk` objects. `make_sdk_embedder()` factory creates an embedder using Databricks SDK against `databricks-gte-large-en`. `format_context()` method produces markdown-formatted LLM context strings. Skipped Mosaic AI Vector Search — unnecessary for 20 chunks.
- [x] Test retrieval quality for each gap analysis query type. Tested on serverless via `execute_code` with all 4 query types. Interest-holding gaps: all 3 target customer profiles (Anderson 0.652, Chen 0.644, Rodriguez 0.620) plus Renewable Energy Trends (0.667) in top 5. Risk alignment: Chen (0.568), Moderate Risk Guide (0.562), Rodriguez (0.550) in top 5. Data quality: all 3 customer profiles in top 5 (0.609, 0.599, 0.596). Investment themes: Retail Disruption (0.593), Tech Sector Analysis (0.567), Investment Guide (0.550) in top 5. All target documents retrieved correctly.
- [x] Retrieval quality is sufficient with pre-computed chunks. No re-chunking needed. All target documents appear in top-5 for every query type with cosine similarity scores 0.55–0.67.
- [x] Codified into `src/semantic_auth/retrieval.py`. Module exposes `DocumentRetrieval` class with `query(text, top_k)` returning `RetrievedChunk` dataclass objects and `format_context(text, top_k)` returning markdown strings. Backend is in-memory cosine similarity behind a stable interface — swap to Neo4j vector index or Mosaic AI Vector Search by replacing internals without changing callers.

### Phase 3: LLM Synthesis

Replace the Supervisor Agent with a single synthesis step that combines structured data and document chunks into a gap analysis prompt and sends it to `databricks-claude-sonnet-4-6`. Use `query_serving_endpoint` to prototype the prompt template interactively before codifying it into a module.

- [x] Design the prompt template. Three-section user message: structured data (labeled as factual portfolio/account data), document excerpts (labeled as qualitative profile and research content), analysis instructions (adapted from `mas_client.py` prompts). System prompt establishes the financial analyst / graph augmentation role. Prototyped and validated via `execute_code` on serverless against `databricks-claude-sonnet-4-6`. Key SDK finding: the serverless runtime's SDK version requires `ChatMessage` / `ChatMessageRole` objects (not plain dicts) for `serving_endpoints.query()`.
- [x] Build synthesis module (`src/semantic_auth/synthesis.py`). `GapAnalysisSynthesizer` class with 5 per-query-type methods mirroring the workshop's `mas_client.py`: `analyze_interest_holding_gaps()`, `analyze_risk_alignment()`, `analyze_data_quality_gaps()`, `extract_investment_themes()`, `run_comprehensive_analysis()`. Each method assembles the right structured data + retrieved docs into a prompt. `make_sdk_caller()` factory creates an `LLMCaller` using `ChatMessage`/`ChatMessageRole`. `fetch_gap_analysis()` convenience function is a drop-in replacement for the workshop's `mas_client.fetch_gap_analysis()`. All five Supervisor Agent prompt texts adapted as module-level constants.
- [x] Test synthesis output against known gap analysis results. Comprehensive query produced 14,630 chars in 248s. Correctly identifies: (1) James Anderson holds 100% Technology but expresses renewable energy interest — "expressed interest in expanding his portfolio to include renewable energy stocks"; (2) Maria Rodriguez has ESG/sustainable investing interest not reflected in holdings — "expressed particular interest in...ESG funds and companies with strong sustainability practices"; (3) Robert Chen's aggressive growth profile identified. Output includes all 4 parts (interest-holding gaps, missing relationships, missing attributes, investment themes) with evidence tables and document quotes.
### Phase 4: DSPy Integration

Wire the synthesis output into the existing DSPy analyzers, replacing the `DatabricksResponsesLM` and Supervisor Agent dependency with `databricks-claude-sonnet-4-6` as a standard chat-completions endpoint. The synthesis step and the analyzers use separate DSPy LM configurations: the synthesis call uses a higher max token limit for context assembly, while the analyzers use tighter temperature for structured output extraction. Run the full pipeline via `databricks-job-runner`: build the `semantic_auth` wheel, submit `run_semantic_auth.py` to serverless, and inspect output via the runner's `logs` subcommand.

- [x] Scaffold `databricks-job-runner` integration in semantic-auth: `databricks-job-runner` already declared as an editable dep in `pyproject.toml`. Created `cli/__init__.py` with `Runner(run_name_prefix="semantic_auth", wheel_package="semantic_auth", build_params=...)` — `build_params` forwards `SOURCE_CATALOG`, `SOURCE_SCHEMA`, `CATALOG_NAME`, `SCHEMA_NAME`, `VOLUME_NAME`, `LLM_ENDPOINT`, `EMBEDDING_ENDPOINT`, `WAREHOUSE_ID` as CLI flags. Created `cli/__main__.py` entry point. Created `agent_modules/test_hello.py` sanity script. Updated `.env.example` with `DATABRICKS_COMPUTE_MODE=serverless`, `DATABRICKS_SERVERLESS_ENV_VERSION=3`, `DATABRICKS_WORKSPACE_DIR`, `DATABRICKS_VOLUME_PATH`, plus all semantic-auth extras.
- [x] Configure DSPy with a standard `dspy.LM` pointing at `databricks-claude-sonnet-4-6` for the analyzer calls. Uses the LiteLLM `databricks/` provider which speaks the OpenAI-compatible chat completions API — no custom `BaseLM` adapter needed. `run_semantic_auth.py` sets `dspy.LM("databricks/{endpoint}", api_key=wc.config.token, api_base=wc.config.host, temperature=0.1, max_tokens=4000)`. The synthesis step uses the separate `make_sdk_caller(max_tokens=8192)` from the synthesis module (higher token limit for context assembly).
- [x] Vendor the DSPy modules from graph-augmented-ai-workshop into semantic-auth: `schemas.py` (Pydantic output models, unchanged), `signatures.py` (DSPy signatures, import fix), `analyzers.py` (four analyzer modules + `GraphAugmentationAnalyzer` with `dspy.Parallel`, import fix), `reporting.py` (result formatting + validation harness, import fix). Dropped `lm.py` (`DatabricksResponsesLM` replaced by standard `dspy.LM`) and `supervisor_client.py` (replaced by `semantic_auth.synthesis`). All imports updated from `augmentation_agent.*` to `semantic_auth.*`.
- [x] Build the pipeline entry point `agent_modules/run_semantic_auth.py`: parses forwarded `build_params` flags via argparse, authenticates via `WorkspaceClient`, configures DSPy with standard `dspy.LM`, sets up structured data access (SDK or Spark executor), loads document embeddings from UC volume, runs `fetch_gap_analysis()` for synthesis, feeds result into `GraphAugmentationAnalyzer` (4 concurrent analyses via `dspy.Parallel`), prints `AugmentationResponse` via `print_response_summary()`. Supports `--enable-tracing` flag for MLflow. Runner auto-attaches `semantic_auth` wheel when script name matches `wheel_package`.
- [x] Run the full pipeline via the runner: `python -m cli upload --wheel` builds and uploads the `semantic_auth` wheel to `<DATABRICKS_VOLUME_PATH>/wheels/`, `python -m cli upload run_semantic_auth.py` uploads the runner script, and `python -m cli submit run_semantic_auth.py` submits to serverless with the wheel auto-attached as an `Environment.dependencies` entry and waits for completion. Retrieve output via `python -m cli logs`. All four parallel analyzers produce structured output: `SuggestedNode` objects (FINANCIAL_GOAL, INVESTMENT_INTEREST, INVESTMENT_THEME), `SuggestedRelationship` objects (INTERESTED_IN, HAS_GOAL), `SuggestedAttribute` objects (Customer.occupation, Customer.investment_philosophy, Customer.life_stage, etc.), and `InvestmentTheme` objects (10 themes). Run ID 649594017312543, 17 suggestions total, 13 high confidence. Fixes applied during integration: moved `databricks-job-runner` to `[cli]` extra, moved dspy/pydantic to core deps, extracted serverless auth token from `wc.config.authenticate()`, preferred Spark executor over SDK, increased SDK HTTP timeout to 600s, set `api_base` to `{host}/serving-endpoints` for LiteLLM, updated `build_params` signature for current job runner API.
- [ ] Enable MLflow tracing (`mlflow.dspy.autolog()`) via `--enable-tracing` flag and verify that traces capture the full pipeline: synthesis call, four parallel analyzer calls, and structured output extraction. (Deferred — flag exists but not yet tested on serverless. See NEXT.md.)
- [x] Spot-check the `AugmentationResponse` output against the known gap analysis results. Pipeline correctly identifies: (1) James Anderson — 100% Technology holdings vs. stated renewable energy interest, (2) Maria Rodriguez — ESG/sustainable investing interest not reflected in portfolio, (3) Robert Chen — aggressive growth profile. Suggestion categories match workshop Lab 7: new node types (FINANCIAL_GOAL, INVESTMENT_INTEREST), relationships (INTERESTED_IN, HAS_GOAL), attributes (occupation, investment_philosophy, life_stage). 3 of 4 analyzers hit `max_tokens=4000` truncation but still produced valid structured output. See NEXT.md for truncation fix and further tuning recommendations.

### Phase 5: Packaging and Documentation

Finalize the project for repeatable runs via `databricks-job-runner`.

- [x] Finalize `.env.example` so that a fresh checkout only needs values filled in: `DATABRICKS_PROFILE`, `DATABRICKS_COMPUTE_MODE=serverless`, `DATABRICKS_SERVERLESS_ENV_VERSION=3`, `DATABRICKS_WORKSPACE_DIR`, `DATABRICKS_VOLUME_PATH`, and all semantic-auth extras (`SOURCE_CATALOG`, `SOURCE_SCHEMA`, `CATALOG_NAME`, `SCHEMA_NAME`, `VOLUME_NAME`, `LLM_ENDPOINT`, `EMBEDDING_ENDPOINT`, `WAREHOUSE_ID`). Neo4j vector search not used (Phase 2 chose in-memory cosine similarity), so `NEO4J_*` keys omitted.
- [x] Verify `build_params` in `cli/__init__.py` forwards every semantic-auth extra as a CLI flag so `agent_modules/run_semantic_auth.py` is fully `.env`-driven (no hardcoded values in submitted scripts). All 8 keys mapped: `SOURCE_CATALOG`, `SOURCE_SCHEMA`, `CATALOG_NAME`, `SCHEMA_NAME`, `VOLUME_NAME`, `LLM_ENDPOINT`, `EMBEDDING_ENDPOINT`, `WAREHOUSE_ID`.
- [ ] Write a README documenting the workflow: prerequisites (Databricks profile configured via `databricks-config` or env vars, foundation model endpoints enabled, catalog provisioned from Phase 0), local setup (copy `.env.example` → `.env`, fill in values, `uv sync`), typical commands (`python -m cli upload --wheel`, `python -m cli submit run_semantic_auth.py`, `python -m cli logs`, `python -m cli clean --yes`), and how to switch between serverless and classic cluster via `DATABRICKS_COMPUTE_MODE`.
- [ ] Document how the project relates to graph-augmented-ai-workshop: which labs it replaces (Lab 5 Genie + Knowledge Agent, Lab 6 Supervisor Agent, Lab 7 Supervisor Agent-backed DSPy), which it reuses (the Neo4j export from Lab 4 → `neo4j_augmentation_demo.raw_data`), and what future phases will add (writing enrichment results back to Neo4j).
