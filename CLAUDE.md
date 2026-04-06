# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Databricks Skills

This project has Databricks Claude Code skills installed in `.claude/skills/`. Skills teach patterns and pair with the Databricks MCP server for execution. Use the matching skill when working in these areas:

- **SQL & Analytics**: `databricks-dbsql` (advanced SQL, materialized views), `databricks-unity-catalog` (system tables, volumes)
- **Development & Deployment**: `databricks-bundles` (DABs, multi-env deploy), `databricks-python-sdk` (SDK, Connect, REST API), `databricks-execution-compute` (run code, manage clusters), `databricks-config` (workspace switching, auth)
- **Data Engineering**: `databricks-spark-declarative-pipelines` (streaming tables, CDC, Auto Loader)
- **AI & Agents**: `databricks-ai-functions` (SQL AI functions, RAG), `databricks-agent-bricks` (Knowledge Assistants, Genie, Supervisor), `databricks-vector-search` (vector indexes, similarity search), `databricks-model-serving` (deploy models/agents)
- **General**: `databricks-docs` (authoritative docs lookup when other skills don't cover it)

To install additional skills: `/Users/ryanknight/projects/databricks/ai-dev-kit/databricks-skills/install_skills.sh <skill-name>`

### Workspace Profile

Default profile is `azure-rk-knight`. Use `manage_workspace(action="switch", profile="azure-rk-knight")` at session start. Check status with `manage_workspace(action="status")`. If token expired, re-auth via `manage_workspace(action="login", host="https://adb-1098933906466604.4.azuredatabricks.net/")`.

## What This Project Does

semantic-auth is a graph enrichment pipeline that replaces Databricks AI/BI services (Genie Space, Knowledge Agent, Multi-Agent Supervisor) with direct LLM calls to foundation model endpoints. It analyzes structured customer portfolio data (Delta tables) and unstructured HTML documents to suggest new nodes, relationships, attributes, and investment themes for a Neo4j knowledge graph.

## Build & Development Commands

```bash
uv sync                              # Install core dependencies
uv sync --extra dev                   # Install dev tools (pytest, ruff)
uv sync --extra enrichment            # Install DSPy/MLflow/Pydantic for analysis
pytest                                # Run tests
ruff check .                          # Lint
ruff format .                         # Format
python -m cli upload --wheel          # Build wheel and upload to Databricks volume
python -m cli submit run_semantic_auth.py  # Submit pipeline to Databricks serverless
```

The `cli` module wraps `databricks-job-runner` — it reads `.env` and forwards values as CLI flags to the pipeline entry point (`agent_modules/run_semantic_auth.py`).

## Architecture

Four layers, each depending only on the layer below:

**Config** (`config.py`) — `Config` dataclass loaded from environment variables. Source catalog/schema (workshop data), target catalog/schema (enrichment artifacts), LLM/embedding endpoint names.

**Data Access** — Two independent data sources:
- `structured_data.py` — `StructuredDataAccess` runs SQL against 14 Neo4j-exported Delta tables. Uses either a Spark session (on-cluster) or SDK statement execution (local, requires `WAREHOUSE_ID`). Factory functions: `make_spark_executor()`, `make_sdk_executor()`.
- `retrieval.py` — `DocumentRetrieval` loads pre-computed embeddings from a JSON file in a UC volume and performs in-memory cosine similarity search (~20 chunks). Factory: `make_sdk_embedder()`.

**Synthesis** (`synthesis.py`) — `GapAnalysisSynthesizer` combines structured context + retrieved documents into LLM prompts. Five query types: `interest_holding_gaps`, `risk_alignment`, `data_quality_gaps`, `investment_themes`, `comprehensive`. Uses `WorkspaceClient.serving_endpoints.query()` via `make_sdk_caller()`. The `fetch_gap_analysis()` function is a drop-in replacement for the workshop's `mas_client.fetch_gap_analysis()`.

**Analysis & Output**:
- `schemas.py` — Pure Pydantic models (no DSPy dependency): `SuggestedNode`, `SuggestedRelationship`, `SuggestedAttribute`, `AugmentationResponse`.
- `signatures.py` — DSPy declarative signatures referencing schema classes.
- `analyzers.py` — Four `dspy.ChainOfThought` analyzers (`InvestmentThemes`, `NewEntities`, `MissingAttributes`, `ImpliedRelationships`) orchestrated concurrently via `dspy.Parallel` in `GraphAugmentationAnalyzer`.
- `reporting.py` — Pretty-print helpers and `ValidationHarness` for PASS/FAIL tracking.

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
- `SOURCE_CATALOG` / `SOURCE_SCHEMA` — workshop Delta tables
- `CATALOG_NAME` / `SCHEMA_NAME` / `VOLUME_NAME` — enrichment artifacts
- `LLM_ENDPOINT` / `EMBEDDING_ENDPOINT` — model serving endpoints
- `WAREHOUSE_ID` — required for local SDK-based SQL execution (not needed on-cluster)
