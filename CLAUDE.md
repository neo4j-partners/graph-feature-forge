# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Web Fetching

When fetching web content (documentation pages, llms.txt indexes, etc.), use the Firecrawl MCP tools (`firecrawl_scrape`, `firecrawl_search`) instead of WebFetch. If Firecrawl MCP tools are not available, ask the user to install the Firecrawl MCP server before proceeding.

## Databricks Skills

This project has Databricks Claude Code skills installed in `.claude/skills/`. Skills teach patterns and pair with the Databricks MCP server for execution. Use the matching skill when working in these areas:

- **SQL & Analytics**: `databricks-dbsql` (advanced SQL, materialized views), `databricks-unity-catalog` (system tables, volumes)
- **Development & Deployment**: `databricks-bundles` (DABs, multi-env deploy), `databricks-python-sdk` (SDK, Connect, REST API), `databricks-execution-compute` (run code, manage clusters), `databricks-config` (workspace switching, auth)
- **Data Engineering**: `databricks-spark-declarative-pipelines` (streaming tables, CDC, Auto Loader)
- **AI & Agents**: `databricks-ai-functions` (SQL AI functions, RAG), `databricks-agent-bricks` (Knowledge Assistants, Genie, Supervisor), `databricks-vector-search` (vector indexes, similarity search), `databricks-model-serving` (deploy models/agents)
- **General**: `databricks-docs` (authoritative docs lookup when other skills don't cover it)

To install additional skills ask the user.

### Workspace Profile

ask the user for default profile. Then use `manage_workspace(action="switch", profile="[DEFAULT_PROFILE]")` at session start. Check status with `manage_workspace(action="status")`. If token expired ask user to re-authenticate.

## What This Project Does

graph_feature_forge is a graph enrichment pipeline that replaces Databricks AI/BI services (Genie Space, Knowledge Agent, Multi-Agent Supervisor) with direct LLM calls to foundation model endpoints. It analyzes structured customer portfolio data (Delta tables) and unstructured HTML documents to suggest new nodes, relationships, attributes, and investment themes for a Neo4j knowledge graph.

## Build & Development Commands

```bash
uv sync                              # Install core dependencies
uv sync --extra dev                   # Install dev tools (pytest, ruff)
uv sync --extra cli                   # Install CLI (databricks-job-runner)
uv sync --extra enrichment            # Install DSPy/MLflow/Pydantic for analysis
uv run pytest                         # Run tests
uv run ruff check .                   # Lint
uv run ruff format .                  # Format
uv run python -m cli upload --wheel   # Build wheel and upload to Databricks volume
uv run python -m cli submit run_graph_feature_forge.py  # Submit pipeline to Databricks serverless
uv run python -m cli logs             # View logs from most recent run
uv run python -m cli validate         # Verify uploaded files in remote workspace
uv run python -m cli clean            # Delete remote workspace and job runs
```

The `cli` module wraps `databricks-job-runner` — it reads `.env` and forwards values as CLI flags to the pipeline entry point (`agent_modules/run_graph_feature_forge.py`).

## Architecture

The package is organized into subpackages by concern:

**Top-level** — `config.py` (Config dataclass from env vars), `graph_schema.py` (node/relationship metadata), `reporting.py` (pretty-print and validation).

**`data/`** — Data access and retrieval:
- `structured_data.py` — `StructuredDataAccess` runs SQL against 14 Neo4j-exported Delta tables. Factory functions: `make_spark_executor()`, `make_sdk_executor()`.
- `retrieval.py` — `DocumentRetrieval` (in-memory cosine similarity) and `Neo4jRetrieval` (vector index). Factory: `make_sdk_embedder()`.
- `enrichment_store.py` — Delta-based enrichment log for deduplication and audit trail.

**`analysis/`** — LLM-driven gap analysis and DSPy modules:
- `schemas.py` — Pure Pydantic models (no DSPy dependency): `SuggestedNode`, `SuggestedRelationship`, `SuggestedAttribute`, `AugmentationResponse`.
- `signatures.py` — DSPy declarative signatures referencing schema classes.
- `analyzers.py` — Four `dspy.ChainOfThought` analyzers orchestrated concurrently via `dspy.Parallel` in `GraphAugmentationAnalyzer`.
- `synthesis.py` — `GapAnalysisSynthesizer` combines structured context + retrieved documents into LLM prompts. Drop-in `fetch_gap_analysis()` function.

**`graph/`** — Neo4j graph operations:
- `loading.py` — Create Delta tables from CSVs on UC volume.
- `seeding.py` — Seed Neo4j from Delta tables + embeddings.
- `extraction.py` — Extract Neo4j graph to Delta tables via Spark Connector.
- `writeback.py` — Write enrichment proposals back to Neo4j with provenance.

**`ml/`** — Feature engineering and AutoML (opt-in):
- `feature_engineering.py` — GDS FastRP + Louvain, feature table export, scoring.
- `model_training.py` — Scikit-learn training, holdout simulation, model registration.

## Key Design Decisions

- **Schemas decoupled from DSPy**: `schemas.py` has zero DSPy imports so Pydantic models work across all layers.
- **DSPy uses LiteLLM's `databricks/` provider**: Standard `dspy.LM("databricks/{endpoint}")` — no custom adapter needed.
- **Concurrent analysis via `dspy.Parallel`**: Four analyzers run in worker threads with proper DSPy context propagation.
- **In-memory retrieval**: Corpus is ~20 chunks — cosine similarity without infrastructure. Internals can be swapped to Neo4j vector index or Mosaic AI Vector Search.
- **Pin `pydantic<2.12`**: Serverless runtime runs Python 3.10 which lacks typing_extensions Sentinel support.

## Dependencies

- `databricks-job-runner` is installed from PyPI (`>=0.3.0`).
- `enrichment` extras (DSPy, MLflow, Pydantic, neo4j) are Phase 2+ — not needed for infrastructure/data access work.

## Environment

Copy `.env.example` to `.env`. Key variables:
- `DATABRICKS_COMPUTE_MODE=serverless` — run mode
- `SOURCE_CATALOG` / `SOURCE_SCHEMA` — source Delta tables (Neo4j export)
- `CATALOG_NAME` / `SCHEMA_NAME` / `VOLUME_NAME` — enrichment artifacts
- `LLM_ENDPOINT` / `EMBEDDING_ENDPOINT` — model serving endpoints
- `WAREHOUSE_ID` — required for local SDK-based SQL execution (not needed on-cluster)
