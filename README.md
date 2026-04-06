[![Databricks](https://img.shields.io/badge/Databricks-Solution_Accelerator-FF3621?style=for-the-badge&logo=databricks)](https://databricks.com)
[![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Enabled-00A1C9?style=for-the-badge)](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
[![Neo4j](https://img.shields.io/badge/Neo4j-Partner-4581C3?style=for-the-badge&logo=neo4j)](https://neo4j.com/partners/databricks/)

# Semantic Auth: Graph Enrichment Pipeline

A graph enrichment pipeline that replaces Databricks AI/BI services (Genie Space, Knowledge Agent, Multi-Agent Supervisor) with direct LLM calls to foundation model endpoints. It analyzes structured customer portfolio data (Delta tables) and unstructured HTML documents to suggest new nodes, relationships, attributes, and investment themes for a Neo4j knowledge graph.

## Architecture

```
Delta Lake Tables (14 tables)          UC Volume (HTML docs + embeddings)
        │                                        │
        ▼                                        ▼
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

## Prerequisites

1. **Databricks workspace** with Unity Catalog enabled
2. **Neo4j database** (Neo4j AuraDB recommended)
3. Source Delta tables and document volumes from the [Graph Augmented AI Workshop](https://github.com/databricks-industry-solutions/graph-enrichment)
4. Foundation model serving endpoints for LLM and embeddings

## Quick Start

```bash
# Install dependencies
uv sync

# Copy environment template and fill in values
cp .env.example .env

# Run tests
uv sync --extra dev
pytest

# Build and upload wheel to Databricks
python -m cli upload --wheel

# Submit pipeline to Databricks serverless
python -m cli submit run_semantic_auth.py
```

## Environment

Copy `.env.example` to `.env`. Key variables:

| Variable | Description |
|----------|-------------|
| `DATABRICKS_COMPUTE_MODE` | Run mode (`serverless`) |
| `SOURCE_CATALOG` / `SOURCE_SCHEMA` | Workshop Delta tables |
| `CATALOG_NAME` / `SCHEMA_NAME` / `VOLUME_NAME` | Enrichment artifacts |
| `LLM_ENDPOINT` / `EMBEDDING_ENDPOINT` | Model serving endpoints |
| `WAREHOUSE_ID` | Required for local SDK-based SQL execution (not needed on-cluster) |

## Project Structure

```
semantic-auth/
├── src/semantic_auth/
│   ├── config.py              # Config dataclass from env vars
│   ├── structured_data.py     # SQL against Neo4j-exported Delta tables
│   ├── retrieval.py           # In-memory cosine similarity retrieval
│   ├── synthesis.py           # LLM gap analysis via foundation models
│   ├── schemas.py             # Pydantic models (no DSPy dependency)
│   ├── signatures.py          # DSPy declarative signatures
│   ├── analyzers.py           # Four concurrent ChainOfThought analyzers
│   └── reporting.py           # Pretty-print and validation harness
├── agent_modules/
│   └── run_semantic_auth.py   # Pipeline entry point for Databricks jobs
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
| python-dotenv | Environment variable loading | BSD-3 | https://github.com/theskumar/python-dotenv |
| mlflow | ML experiment tracking | Apache 2.0 | https://github.com/mlflow/mlflow |
| databricks-job-runner | Databricks job submission CLI | Apache 2.0 | https://pypi.org/project/databricks-job-runner/ |

&copy; 2026 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source). All included or referenced third party libraries are subject to the licenses set forth above.
