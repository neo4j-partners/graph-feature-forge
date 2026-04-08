# Graph Feature Forge

A graph feature engineering pipeline that combines LLM-driven graph enrichment with Neo4j GDS algorithms to produce ML-ready feature tables on Databricks. It uses direct LLM calls to Databricks Model Serving endpoints for graph enrichment via foundation models.

### Enrichment Pipeline

The enrichment pipeline analyzes structured customer portfolio data from Delta tables and unstructured HTML documents using foundation model endpoints and four concurrent DSPy analyzers to discover new nodes, relationships, attributes, and investment themes. Proposals are deduplicated against a Delta-based enrichment log and dual-written to Neo4j, compounding the graph's knowledge with each run.

### Feature Engineering Stage

The enrichment pipeline can only analyze customers that have profile documents, and in this dataset only 3 of 103 customers do. The feature engineering stage closes that gap by extracting structural signal from the full graph topology, which connects all customers regardless of document coverage. FastRP encodes each customer's pattern of holdings, accounts, and connected companies into a 128-dimensional vector, so customers with similar portfolios get similar embeddings. AutoML then learns which structural patterns correspond to which risk profiles and predicts them for the undocumented customers.

The feature engineering stage:

- Projects the enriched graph in Neo4j GDS and computes FastRP embeddings and Louvain community detection
- Exports graph features to Delta tables alongside tabular features
- Trains classifiers with Databricks AutoML on the combined feature set
- Scores the undocumented customers and writes predictions back to Neo4j so the next enrichment cycle has richer context for everyone
- Three standalone notebooks demonstrate the full ML lifecycle with a three-way model comparison (FastRP-only, FastRP + Louvain, tabular-only baseline)

The entire enrichment pipeline runs as automated Databricks jobs via a CLI that wraps `databricks-job-runner`. Data upload, Delta table creation, Neo4j seeding, and enrichment execute on serverless compute with a single `./run_pipeline.sh` command. The GDS feature engineering notebooks run on Databricks Runtime 17.x LTS ML clusters with the Neo4j Spark Connector.

The project demonstrates these Databricks services end-to-end:

- **Unity Catalog** — Governs Delta tables, UC Volumes for file storage, and the MLflow model registry with Champion/Challenger aliases across the entire pipeline
- **MLflow Experiment Tracking** — Three parallel experiments compare graph-augmented vs. tabular-only classifiers, with model registration, feature importance analysis, and automated Champion promotion
- **MLflow DSPy Autolog Tracing** — Optional tracing of all DSPy LLM calls through MLflow for observability and debugging of the enrichment pipeline

```
                          Raw Data (CSV, HTML, embeddings)
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │           UC Volume / Delta           │
                    │   14 tables (7 node + 7 relationship) │
                    └──────────────────┬───────────────────┘
                                       │  seed
                                       ▼
                    ┌──────────────────────────────────────┐
                    │        Neo4j Knowledge Graph          │
                    └───────┬──────────────────────┬───────┘
                            │                      │
               ┌────────────▼───────────┐ ┌────────▼───────────────┐
               │    Enrichment Loop     │ │  Feature Engineering   │
               │                        │ │                        │
               │  Structured Data (SQL) │ │  GDS Projection        │
               │  + Doc Retrieval       │ │  FastRP (128-dim)      │
               │        │               │ │  Louvain Communities   │
               │        ▼               │ │        │               │
               │  LLM Gap Synthesis     │ │        ▼               │
               │        │               │ │  Delta Feature Table   │
               │        ▼               │ │        │               │
               │  DSPy Analyzers (x4)   │ │        ▼               │
               │        │               │ │  AutoML Classifier     │
               │        ▼               │ │        │               │
               │  Dedup + Dual-Write    │ │  Score + Write Back    │
               └────────┬───────────────┘ └────────┬───────────────┘
                        │                          │
                        └────► Neo4j ◄─────────────┘
                          (each cycle compounds)
```

## Prerequisites

1. **Databricks workspace** with Unity Catalog enabled
2. **Neo4j database** (Neo4j AuraDB recommended)
3. Foundation model serving endpoints for LLM and embeddings
4. **Neo4j GDS plugin** (AuraDS recommended) for feature engineering notebooks
5. **Databricks Runtime 17.x LTS ML** for notebooks (AutoML removed as built-in in 18.0+)
6. **Neo4j Spark Connector** (`org.neo4j:neo4j-connector-apache-spark_2.12:5.x`) as a cluster library for notebooks

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

# Install GDS feature engineering dependencies
uv sync --extra gds
```

### Option A: One-command script

```bash
# Run all steps (upload → load → seed → enrich)
./run_pipeline.sh

# Run a single phase
./run_pipeline.sh load     # Phase 1: Upload data + create Delta tables
./run_pipeline.sh seed     # Phase 2: Seed Neo4j from Delta tables
./run_pipeline.sh enrich   # Phase 3: Run enrichment pipeline

# Run GDS feature engineering (requires ML Runtime cluster)
./run_pipeline.sh gds
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

### Pipeline Phases

The enrichment pipeline runs as a sequence of Databricks jobs. Each phase builds on the previous one, and the enrichment loop is designed to run repeatedly — each cycle discovers new proposals from an increasingly rich graph.

1. **Data Ingestion** (`graph/loading.py`, `agent_modules/load_data.py`) — Upload raw CSV, HTML, and embedding files to a UC Volume. Create 14 Delta tables (7 node tables, 7 relationship tables) from the CSVs. Idempotent — skips if tables already exist.

2. **Neo4j Seeding** (`graph/seeding.py`, `agent_modules/seed_neo4j.py`) — Seed the Neo4j knowledge graph from Delta tables using batched MERGE statements. Load pre-computed document chunk embeddings and create a vector index for retrieval.

3. **Enrichment Extraction** (`graph/extraction.py`) — Dynamically discover all node labels and relationship types in Neo4j, extract enrichment-only data back to Delta tables (skips base types to avoid duplication). Uses the Neo4j Spark Connector.

4. **Gap Analysis Synthesis** (`data/structured_data.py`, `data/retrieval.py`, `analysis/synthesis.py`) — SQL queries against Delta tables for structured context. Neo4j vector index for document retrieval. The LLM synthesizes both into a gap analysis via foundation model endpoints. Prior enrichments from the enrichment log are injected as additional context.

5. **DSPy Analysis** (`analysis/schemas.py`, `analysis/signatures.py`, `analysis/analyzers.py`) — Four concurrent `ChainOfThought` analyzers via `dspy.Parallel`: InvestmentThemes, NewEntities, MissingAttributes, ImpliedRelationships. Schema-level suggestions are resolved to concrete instance proposals by the `InstanceResolver`.

6. **Deduplication and Writeback** (`data/enrichment_store.py`, `graph/writeback.py`) — Instance proposals are deduplicated against the `enrichment_log` Delta table. HIGH-confidence proposals are dual-written: idempotent MERGE to Neo4j with provenance properties, and INSERT to the enrichment log for future deduplication and context injection.

7. **GDS Feature Engineering** (`ml/feature_engineering.py`) — Project the enriched graph in GDS. Compute FastRP embeddings (128 dimensions) and Louvain community detection. Export to a Delta feature table. Score unlabeled customers using a registered MLflow Champion model and write predictions back to Neo4j. This step is opt-in via `ENABLE_GDS_FEATURES=true`.

### GDS Feature Engineering

The GDS feature engineering stage runs as three sequential Databricks jobs via `./run_pipeline.sh gds`. Each job has a corresponding entry point in `agent_modules/` and an interactive notebook in `notebooks/` for exploration.

| Entry point | Notebook | What it does | Depends on |
|-------------|----------|-------------|------------|
| `gds_fastrp_features.py` | `notebooks/gds_fastrp_features.py` | Projects portfolio graph in GDS, computes 128-dim FastRP embeddings, exports to Delta, trains AutoML classifier, scores held-out customers, writes predictions to Neo4j. Registers model with Champion alias. | Neo4j Aura with GDS |
| `gds_community_features.py` | `notebooks/gds_community_features.py` | Adds Louvain community detection as a categorical feature. Retrains AutoML with combined features, promotes Champion if F1 improves. Runs kNN for nearest-neighbor visualization. | `gds_fastrp_features` |
| `gds_baseline_comparison.py` | `notebooks/gds_baseline_comparison.py` | Trains tabular-only model (annual_income, credit_score). Produces three-way MLflow comparison and feature importance analysis. | `gds_fastrp_features` |

The entry points run on Databricks Runtime 17.x LTS ML clusters (AutoML removed as built-in in 18.0+). The notebooks remain for interactive exploration in the Databricks notebook UI.

Each notebook writes to a separate MLflow experiment for side-by-side comparison:

| Experiment | Features used |
|------------|--------------|
| `/Shared/graph-feature-forge/fastrp_risk_classification` | 128 FastRP dimensions + tabular |
| `/Shared/graph-feature-forge/fastrp_louvain_risk_classification` | 128 FastRP dimensions + community_id + tabular |
| `/Shared/graph-feature-forge/tabular_only_baseline` | Tabular only (annual_income, credit_score) |

### Automated Job Deployment

The `cli/` module wraps `databricks-job-runner` — it reads `.env` and forwards values as CLI flags to the pipeline entry points. The `run_pipeline.sh` script orchestrates the pipeline:

```bash
./run_pipeline.sh          # All phases: load → seed → enrich
./run_pipeline.sh load     # Phase 1: Upload data + create Delta tables
./run_pipeline.sh seed     # Phase 2: Seed Neo4j from Delta tables
./run_pipeline.sh enrich   # Phase 3: Run enrichment pipeline
./run_pipeline.sh gds      # Phase 4: GDS feature engineering (ML Runtime cluster)
./run_pipeline.sh html     # Generate HTML documents + embeddings via LLM endpoint
```

Each phase builds the wheel, uploads the entry point, and submits a job. The enrichment entry points run on serverless compute. The GDS entry points require a Databricks Runtime 17.x LTS ML cluster with the Neo4j Spark Connector.

| Entry point | Phase | What it does |
|-------------|-------|-------------|
| `load_data.py` | 1 | Create 14 Delta tables from CSV files on the UC Volume |
| `seed_neo4j.py` | 2 | Seed Neo4j nodes, relationships, document graph, and vector index |
| `run_graph_feature_forge.py` | 3 | Full enrichment pipeline: extract → synthesize → analyze → dedup → write |
| `generate_html.py` | html | Generate HTML docs + embeddings via LLM/embedding endpoints on-cluster |
| `gds_fastrp_features.py` | 4a | FastRP → Delta export → holdout → AutoML → register Champion → score → Neo4j writeback |
| `gds_community_features.py` | 4b | FastRP + Louvain → retrain AutoML → promote Champion if F1 improves → kNN analysis |
| `gds_baseline_comparison.py` | 4c | Tabular-only AutoML → three-way MLflow comparison → feature importance |

Neo4j credentials are protected via a Databricks secret scope. Run `./create_secrets.sh` to provision the scope from `.env` values.

## Environment

Copy `.env.example` to `.env`. Key variables:

| Variable | Description |
|----------|-------------|
| `DATABRICKS_COMPUTE_MODE` | Run mode (`serverless`) |
| `SOURCE_CATALOG` / `SOURCE_SCHEMA` | Workshop Delta tables |
| `CATALOG_NAME` / `SCHEMA_NAME` / `VOLUME_NAME` | Enrichment artifacts |
| `LLM_ENDPOINT` / `EMBEDDING_ENDPOINT` | Model serving endpoints |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Neo4j connection (required for seeding and enrichment) |
| `NEO4J_DATABASE` | Neo4j database name (default: `neo4j`) |
| `WAREHOUSE_ID` | Required for local SDK-based SQL execution (not needed on-cluster) |
| `DATABRICKS_SECRET_SCOPE` | Databricks secret scope for Neo4j credentials |

## Project Structure

```
graph-feature-forge/
├── data/
│   ├── csv/                   # 7 CSV files (customers, banks, accounts, etc.)
│   ├── html/                  # HTML documents (profiles, analyses, guides)
│   ├── embeddings/            # Pre-computed 1024-dim document chunk embeddings
│   ├── csv_generator/         # Part 1: Synthetic CSV data generator (Faker + Pydantic)
│   └── html_generator/        # Part 2: LLM-generated HTML + embedding generator
├── src/graph_feature_forge/
│   ├── config.py              # Config dataclass from env vars
│   ├── graph_schema.py        # Shared node/relationship metadata registry
│   ├── reporting.py           # Pretty-print and validation harness
│   ├── data/                  # Data access & retrieval
│   │   ├── structured_data.py # SQL against Delta tables
│   │   ├── retrieval.py       # Document retrieval (in-memory or Neo4j vector index)
│   │   └── enrichment_store.py# Delta-based enrichment log with deduplication
│   ├── analysis/              # LLM-driven gap analysis & DSPy modules
│   │   ├── schemas.py         # Pydantic models (no DSPy dependency)
│   │   ├── signatures.py      # DSPy declarative signatures
│   │   ├── analyzers.py       # Four concurrent ChainOfThought analyzers
│   │   └── synthesis.py       # LLM gap analysis via foundation models
│   ├── graph/                 # Neo4j graph operations
│   │   ├── loading.py         # Create Delta tables from CSVs on UC volume
│   │   ├── seeding.py         # Seed Neo4j from Delta tables + embeddings
│   │   ├── extraction.py      # Extract Neo4j graph to Delta tables
│   │   └── writeback.py       # Write enrichment proposals back to Neo4j
│   └── ml/                    # Feature engineering & AutoML (opt-in)
│       ├── feature_engineering.py # GDS FastRP + Louvain + feature table export
│       └── automl_training.py # AutoML training, holdout, model registration
├── notebooks/
│   ├── gds_fastrp_features.py       # FastRP → AutoML → Neo4j writeback
│   ├── gds_community_features.py    # + Louvain community detection
│   └── gds_baseline_comparison.py   # Tabular-only baseline comparison
├── agent_modules/
│   ├── load_data.py                 # Create Delta tables from raw CSVs
│   ├── seed_neo4j.py                # Seed Neo4j from Delta tables
│   ├── run_graph_feature_forge.py   # Full enrichment pipeline
│   ├── generate_html.py             # Generate HTML docs + embeddings via LLM
│   ├── gds_fastrp_features.py       # FastRP → AutoML → Neo4j writeback
│   ├── gds_community_features.py    # + Louvain community detection
│   └── gds_baseline_comparison.py   # Tabular-only baseline comparison
├── cli/                       # Job submission CLI (wraps databricks-job-runner)
├── tests/                     # Unit tests
├── run_pipeline.sh            # One-command pipeline orchestrator
├── create_secrets.sh          # Provision Databricks secret scope from .env
├── pyproject.toml
└── .env.example
```

## License

Apache License 2.0

---

This is a [Neo4j Labs](https://neo4j.com/labs/) project -- community supported, not officially backed by Neo4j. [Community Forum](https://community.neo4j.com) | [GitHub Issues](https://github.com/neo4j-labs/agent-memory/issues)
