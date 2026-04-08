# Data Directory

Raw data files and the generators that produce them.

## Directory Structure

```
data/
├── csv/                  # 7 CSV files (customers, banks, accounts, etc.)
├── html/                 # HTML documents (profiles, analyses, guides)
├── embeddings/           # Pre-computed 1024-dim document chunk embeddings
├── csv_generator/        # Part 1: Synthetic CSV data generator
└── html_generator/       # Part 2: LLM-generated HTML + embedding generator
```

## Part 1: CSV Generator

Generates all 7 CSV files with realistic financial portfolio data using
Faker and seeded randomness. Runs locally — no Databricks access needed.

```bash
# Install generator dependencies
uv sync --extra generator

# Generate CSVs (default: 500 customers, seed=42)
uv run python -m data.csv_generator

# Override counts via env vars (GEN_ prefix)
GEN_NUM_CUSTOMERS=1000 GEN_NUM_BANKS=100 uv run python -m data.csv_generator
```

Default output: `data/csv/`. Fully reproducible via `GEN_RANDOM_SEED`.

## Part 2: HTML & Embedding Generator

Generates HTML documents using an LLM endpoint and embeds them using the
Databricks gte-large-en endpoint. Reads Part 1 CSVs for entity references.

### Local (dry-run)

Uses templates and random vectors — no Databricks access needed.

```bash
uv run python -m data.html_generator --dry-run
```

### Local (live)

Calls Databricks endpoints directly. Requires `DATABRICKS_HOST` and
`DATABRICKS_TOKEN` environment variables.

```bash
uv run python -m data.html_generator
```

### On Databricks (as a job)

The html_generator is included in the project wheel and can be submitted
as a Databricks job via the CLI, just like other pipeline steps.

```bash
# Build wheel (includes html_generator) and upload
uv run python -m cli upload --wheel
uv run python -m cli upload generate_html.py

# Submit job — reads CSVs from UC volume, writes HTML + embeddings back
uv run python -m cli submit generate_html.py
```

The agent module (`agent_modules/generate_html.py`) reads CSVs from the
UC Volume FUSE path, calls the LLM and embedding endpoints, and writes
results directly back to the volume. `LLM_ENDPOINT` and
`EMBEDDING_ENDPOINT` are passed as job parameters from `.env`.

## Generated Data Summary

| File | Entity | Default Count |
|------|--------|---------------|
| `csv/customers.csv` | Customers | 500 (150 labeled) |
| `csv/banks.csv` | Financial institutions | 50 |
| `csv/companies.csv` | Public companies | 200 |
| `csv/stocks.csv` | Stock price data | 200 |
| `csv/accounts.csv` | Bank accounts | ~900 |
| `csv/portfolio_holdings.csv` | Stock positions | ~900 |
| `csv/transactions.csv` | Account transfers | ~3,700 |
| `html/*.html` | Enrichment documents | ~41 |
| `embeddings/document_chunks_embedded.json` | 1024-dim vectors | ~41 chunks |
