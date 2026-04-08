[![Databricks](https://img.shields.io/badge/Databricks-Solution_Accelerator-FF3621?style=for-the-badge&logo=databricks)](https://databricks.com)
[![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Enabled-00A1C9?style=for-the-badge)](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
[![Neo4j](https://img.shields.io/badge/Neo4j-Partner-4581C3?style=for-the-badge&logo=neo4j)](https://neo4j.com/partners/databricks/)

# Graph Feature Forge: Graph Enrichment Pipeline

A graph enrichment pipeline that replaces Databricks AI/BI services (Genie Space, Knowledge Agent, Multi-Agent Supervisor) with direct LLM calls to foundation model endpoints. It analyzes structured customer portfolio data (Delta tables) and unstructured HTML documents to suggest new nodes, relationships, attributes, and investment themes for a Neo4j knowledge graph.

## Prerequisites

1. **Databricks workspace** with Unity Catalog enabled
2. **Neo4j database** (Neo4j AuraDB recommended)
3. Foundation model serving endpoints for LLM and embeddings

## Quick Start

```bash
# Install dependencies
uv sync

# Copy environment template and fill in values
cp .env.example .env

# Install dev tools and run tests
uv sync --extra dev
uv run pytest

# Install CLI
uv sync --extra cli
```

### Option A: One-command script

```bash
# Run all steps (upload → load → seed → enrich)
./run_pipeline.sh

# Stop after seeding Neo4j (skip enrichment)
./run_pipeline.sh --seed
```

### Option B: Manual steps

```bash
# Upload raw data (CSV, HTML, embeddings) to UC volume
uv run python -m cli upload --data

# Build and upload library wheel + entry points
uv run python -m cli upload --wheel
uv run python -m cli upload load_data.py
uv run python -m cli upload seed_neo4j.py
uv run python -m cli upload run_graph_feature_forge.py

# Create Delta tables from CSVs
uv run python -m cli submit load_data.py

# Seed Neo4j from Delta tables + embeddings
uv run python -m cli submit seed_neo4j.py

# Run enrichment pipeline
uv run python -m cli submit run_graph_feature_forge.py

# View logs from the run
uv run python -m cli logs
```

## Monitoring Runs

```bash
# View logs from the most recent run
uv run python -m cli logs

# View logs from a specific run ID
uv run python -m cli logs <run_id>

# Verify uploaded files exist in the remote workspace
uv run python -m cli validate

# Clean up remote workspace and job runs
uv run python -m cli clean
```

## Architecture

```
Raw Data (CSV, HTML, embeddings)
        │
        ▼  upload --data
UC Volume (data/csv/, data/html/, data/embeddings/)
        │
        ▼  load_data.py
Delta Lake Tables (14 tables)          UC Volume (HTML docs + embeddings)
        │                                        │
        ▼  seed_neo4j.py                         │
Neo4j Knowledge Graph ◄─────────────────────────┘
        │
        ▼  run_graph_feature_forge.py
StructuredDataAccess (SQL)             DocumentRetrieval (cosine similarity)
        │                                        │
        └──────────────┬─────────────────────────┘
                       ▼
          GapAnalysisSynthesizer (LLM)
                       │
                       ▼
          GraphAugmentationAnalyzer (DSPy)
          ┌────────────┼────────────────┐
          ▼            ▼                ▼
  InvestmentThemes  NewEntities  ImpliedRelationships
          ▼            ▼                ▼
          └────────────┼────────────────┘
                       ▼
            AugmentationResponse (Pydantic)
```

Four layers, each depending only on the layer below:

- **Config** (`config.py`) — `Config` dataclass loaded from environment variables
- **Data Access** — `StructuredDataAccess` (SQL against Delta tables) and `DocumentRetrieval` (in-memory cosine similarity over ~20 pre-computed embedding chunks)
- **Synthesis** (`synthesis.py`) — Combines structured context + retrieved documents into LLM prompts via foundation model endpoints
- **Analysis** (`analyzers.py`) — Four DSPy `ChainOfThought` analyzers run concurrently via `dspy.Parallel`

## Environment

Copy `.env.example` to `.env`. Key variables:

| Variable | Description |
|----------|-------------|
| `DATABRICKS_COMPUTE_MODE` | Run mode (`serverless`) |
| `SOURCE_CATALOG` / `SOURCE_SCHEMA` | Workshop Delta tables |
| `CATALOG_NAME` / `SCHEMA_NAME` / `VOLUME_NAME` | Enrichment artifacts |
| `LLM_ENDPOINT` / `EMBEDDING_ENDPOINT` | Model serving endpoints |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Neo4j connection (required for seeding and enrichment) |
| `WAREHOUSE_ID` | Required for local SDK-based SQL execution (not needed on-cluster) |

## Project Structure

```
graph-feature-forge/
├── data/
│   ├── csv/                   # 7 CSV files (customers, banks, accounts, etc.)
│   ├── html/                  # 14 HTML documents (profiles, analyses, guides)
│   └── embeddings/            # Pre-computed 1024-dim document chunk embeddings
├── src/graph_feature_forge/
│   ├── config.py              # Config dataclass from env vars
│   ├── loading.py             # Create Delta tables from CSVs on UC volume
│   ├── seeding.py             # Seed Neo4j from Delta tables + embeddings
│   ├── extraction.py          # Extract Neo4j graph to Delta tables
│   ├── structured_data.py     # SQL against Delta tables
│   ├── retrieval.py           # In-memory cosine similarity retrieval
│   ├── synthesis.py           # LLM gap analysis via foundation models
│   ├── schemas.py             # Pydantic models (no DSPy dependency)
│   ├── signatures.py          # DSPy declarative signatures
│   ├── analyzers.py           # Four concurrent ChainOfThought analyzers
│   ├── reporting.py           # Pretty-print and validation harness
│   └── writeback.py           # Write enrichment proposals back to Neo4j
├── agent_modules/
│   ├── load_data.py           # Create Delta tables from raw CSVs
│   ├── seed_neo4j.py          # Seed Neo4j from Delta tables
│   └── run_graph_feature_forge.py   # Full enrichment pipeline
├── tests/                     # Unit tests
├── cli/                       # Job submission CLI (wraps databricks-job-runner)
├── pyproject.toml
└── .env.example
```

## Project Support

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE.md). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support.

## Third-Party Package Licenses

| library | description | license | source |
|---------|-------------|---------|--------|
| databricks-sdk | Databricks Python SDK | Apache 2.0 | https://github.com/databricks/databricks-sdk-py |
| dspy | Structured reasoning framework | MIT | https://github.com/stanfordnlp/dspy |
| pydantic | Data validation | MIT | https://github.com/pydantic/pydantic |
| mlflow | ML experiment tracking | Apache 2.0 | https://github.com/mlflow/mlflow |
| databricks-job-runner | Databricks job submission CLI | Apache 2.0 | https://pypi.org/project/databricks-job-runner/ |

&copy; 2026 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source). All included or referenced third party libraries are subject to the licenses set forth above.
