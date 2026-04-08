# Plan: Grow Dataset from 100 to 500 Customers

**Date:** 2026-04-08
**Status:** Planning
**Motivation:** Phase 2.5 of sklearn.md needs more labeled data to fairly
evaluate graph features. 30 labeled rows is the root constraint — 500
customers with proportional labels gives the ML pipeline enough data to
determine whether FastRP embeddings and community detection actually add
predictive value.

---

## Current Data Inventory

| File | Rows | Entity | Notes |
|------|------|--------|-------|
| customers.csv | 103 | Customers | 30 labeled (risk_profile), 72 unlabeled |
| accounts.csv | 124 | Bank accounts | ~1.2 accounts per customer |
| banks.csv | 103 | Financial institutions | Seems high — likely a fixed universe |
| transactions.csv | 124 | Transfers between accounts | ~1 per account |
| companies.csv | 103 | Public companies | Investment universe |
| stocks.csv | 103 | Stock price data | One per company |
| portfolio_holdings.csv | 111 | Stock positions in accounts | ~1 per account |
| html/ | 14 | Enrichment documents | Profiles, analysis, guides |
| embeddings/ | 20 chunks | Document embeddings | 1024-dim, gte-large-en |

---

## Target Scale

Grow customers to 500 and scale everything else proportionally (~5x). Some
entities are institutional (banks, companies) and should grow less
aggressively since the real world has fewer banks than customers.

| Entity | Current | Target | Rationale |
|--------|---------|--------|-----------|
| Customers | 103 | 500 | Primary growth target |
| Labeled customers | 30 | ~150 | Keep ~30% labeled across all three risk classes (50 per class) |
| Accounts | 124 | ~700 | 1-3 accounts per customer (checking, savings, investment) |
| Banks | 103 | ~50 | Institutional — fewer unique banks, more customers per bank |
| Transactions | 124 | ~2,500 | 3-5 transactions per account feels realistic |
| Companies | 103 | ~200 | Public company universe — grows slower than customers |
| Stocks | 103 | ~200 | One per company |
| Portfolio holdings | 111 | ~1,500 | 2-4 holdings per investment account |
| HTML documents | 14 | ~40 | More customer profiles, company analyses, sector reports |
| Embedding chunks | 20 | ~60 | Proportional to HTML documents |

---

## What to Build

A data generator package modeled after the pharma_demo generator. That
generator uses Faker for realistic names, Pydantic models for validation,
seeded randomness for reproducibility, and configurable counts via
environment variables. We follow the same pattern but for financial
portfolio data.

### Generator Structure

Two generator packages live inside `data/`:

- **models** — Pydantic models matching each CSV schema (Customer, Account,
  Bank, Transaction, Company, Stock, PortfolioHolding). These define the
  columns and validation rules.
- **generators** — One function per entity type. Each takes a count and a
  seed, and returns a list of model instances. Relationships between
  entities use the IDs from previously generated entities (customers before
  accounts, accounts before transactions, etc.).
- **config** — Pydantic Settings class reading counts and parameters from
  environment variables or a `.env` file, with sensible defaults matching
  the target scale above.
- **html_generator** — Produces enrichment HTML documents (customer profiles,
  company analyses, sector reports) using templates filled with generated
  entity data.
- **embedding_generator** — Takes the generated HTML documents and produces
  the chunked embedding JSON file. This either calls the Databricks
  embedding endpoint or generates placeholder vectors for local testing.
- **main** — Orchestrates everything: generate entities in dependency order,
  write CSVs, generate HTML, generate embeddings, optionally write to
  Neo4j.

### Generation Order (Dependency Chain)

Entities must be generated in this order because later entities reference
earlier ones:

1. **Banks** — no dependencies, generates bank_id values
2. **Companies** — no dependencies, generates company_id values
3. **Stocks** — depends on companies (each stock references a company_id)
4. **Customers** — no dependencies, generates customer_id values and
   assigns risk_profile labels to ~30% of them (evenly split across
   Aggressive, Conservative, Moderate)
5. **Accounts** — depends on customers and banks (each account references
   a customer_id and bank_id)
6. **Portfolio holdings** — depends on accounts and stocks (each holding
   references an account_id and stock_id)
7. **Transactions** — depends on accounts (each transaction has
   from_account_id and to_account_id)
8. **HTML documents** — depends on customers, companies, and banks (profiles
   reference real entity names and IDs)
9. **Embeddings** — depends on HTML documents

### Realistic Data Rules

These rules keep the generated data internally consistent and plausible for
financial portfolio analysis.

**Customers:**
- Use Faker for names, emails, phone numbers, addresses
- Annual income drawn from a log-normal distribution (median ~75K, range
  30K-500K) so most customers are middle-income with a long tail
- Credit scores drawn from a normal distribution (mean 700, std 80,
  clamped 300-850) matching real FICO distributions
- Risk profiles assigned to labeled customers should correlate loosely with
  income and credit score — Aggressive customers tend toward higher income,
  Conservative toward higher credit scores — but with noise so the ML model
  has something non-trivial to learn
- Employment status weighted: 70% Employed, 15% Self-Employed, 10% Retired,
  5% Unemployed
- Registration dates spread over 3 years, dates of birth over 40 years
  (ages 25-65)

**Accounts:**
- Each customer gets 1-3 accounts (weighted: 40% one, 40% two, 20% three)
- Account types: Checking, Savings, Investment — investment accounts are
  required for portfolio holdings
- Balances correlate with customer income (higher income = higher average
  balance) but with variance
- Interest rates vary by account type (checking ~0.01%, savings ~2-4%,
  investment ~0%)

**Banks:**
- Use real-sounding bank names from Faker or a curated list
- Bank types: Commercial, Regional, Credit Union, Savings — weighted toward
  Commercial
- Total assets drawn from a wide range (1B-500B)
- Multiple customers share the same bank, creating the graph relationships
  that GDS community detection should find

**Transactions:**
- Each account has 3-5 transactions
- Amounts drawn from a log-normal distribution (median ~500, range 10-50K)
- Transaction types: Transfer, Payment, Deposit, Withdrawal
- From/to accounts should create a connected graph — some transactions go
  between accounts at the same bank (internal transfers), some across banks
- Dates spread over the past year

**Companies and Stocks:**
- Industries and sectors should be diverse (Technology, Healthcare, Finance,
  Energy, Consumer, Industrial)
- Market caps and revenues should be internally consistent (no company with
  1B revenue and 500B market cap)
- Stock prices should be plausible relative to market cap
- P/E ratios should vary by sector (tech higher, utilities lower)

**Portfolio Holdings:**
- Only investment accounts hold stocks
- Each investment account holds 2-4 different stocks
- Purchase prices should be slightly different from current prices (some
  gains, some losses)
- Portfolio percentages within an account should sum to roughly 100%

**HTML Documents:**
- Generate customer profile pages for 5-10 of the most interesting
  customers (varied risk profiles, high/low income)
- Generate company analysis pages for 10-15 notable companies
- Generate 5-10 sector/market analysis pages
- Generate 5-10 investment strategy and financial planning guides
- All documents should reference real entity names and IDs from the
  generated data so the embedding retrieval pipeline can find relevant
  context

**Embeddings:**
- Chunk HTML documents at 4000 characters with 200-character overlap
  (matching current format)
- Use the gte-large-en embedding model via Databricks endpoint for real
  vectors, or generate random 1024-dim vectors for offline testing
- Output format must match the existing `document_chunks_embedded.json`
  schema exactly (metadata block, documents array, chunks array with
  embeddings)

### Configuration

All counts and distribution parameters should be configurable via
environment variables with defaults matching the target scale. Key
settings:

- Number of customers (default 500)
- Number of banks (default 50)
- Number of companies (default 200)
- Fraction of customers labeled (default 0.3)
- Random seed (default 42) for reproducibility
- Output directory (default `data/`)
- Whether to generate embeddings via API or use placeholders

### What to Borrow from the Pharma Generator

The pharma_demo generator does several things well that we should copy:

- **Pydantic models for every entity** — validates data at generation time,
  catches schema drift early
- **Pydantic Settings for configuration** — clean env var loading with
  typed defaults
- **Seeded randomness throughout** — every generator takes a seed so the
  full dataset is reproducible
- **Dependency-ordered generation** — entities generated in topological
  order so foreign keys always resolve
- **Gaussian distributions for realistic counts** — relationship counts
  (accounts per customer, holdings per account) use normal distributions
  centered on configured averages instead of uniform random
- **Batch writing** — the pharma generator batches Neo4j writes at 1000
  rows; we should batch CSV writes similarly if we ever add Neo4j direct
  loading
- **Ground truth tracking** — pharma tracks fraud ring membership; we track
  which customers are labeled and their true risk profiles

### What Not to Borrow

- **Neo4j writer** — we write CSVs and let the existing pipeline load them
  into Neo4j via the seeding module. No need to duplicate that path.
- **Fraud ring injection** — not applicable. Our labeled data is about risk
  profiles, not fraud detection.
- **OAuth/Keycloak auth** — our Neo4j setup uses simpler auth.

---

## How This Helps the ML Pipeline

The sklearn.md plan (Phase 2.5) identified two problems: too few labeled
rows (30) and too many features (128 FastRP dimensions). Growing to 500
customers with 150 labeled fixes the first problem directly:

- **150 labeled rows** means k=5 cross-validation gives 120 train / 30 test
  per fold — much more stable than the current 20 train / 10 test
- **50 per risk class** means stratified splits have enough samples per
  class to estimate F1 reliably
- **80/20 train/test split** gives 120 train / 30 test overall, instead of
  the current 30 train / 72 test (inverted ratio)
- **More graph edges** from 500 customers means FastRP embeddings capture
  richer structural patterns — community detection should find actual
  clusters instead of noise
- **Feature ratio improves** even without PCA: 130 features on 120 training
  rows is marginal but workable, versus impossible on 30 rows

This dataset growth is a prerequisite for getting a fair answer to the
Phase 2 question: do graph features add predictive value?

---

## Deliverables

1. `data/csv_generator/` — CSV generation package
2. `data/html_generator/` — HTML + embedding generation package
3. `agent_modules/generate_html.py` — Databricks job entry point for HTML generation
4. Regenerated CSV files in `data/csv/` (500 customers, proportional everything)
5. Regenerated HTML files in `data/html/` (~40 documents)
6. Regenerated embedding file in `data/embeddings/` (~60 chunks)

---

## Decisions

- **Regenerate from scratch.** All entity IDs start fresh. Re-seed Neo4j
  after generation.
- **CSV output only.** No Neo4j writer in the generator — the existing
  seeding module loads CSVs into Neo4j.
- **LLM-generated HTML documents.** More realistic for testing the
  retrieval pipeline.
- **Real embedding vectors.** Generated via the Databricks gte-large-en
  endpoint so the retrieval pipeline works end-to-end.

---

## Two-Part Split

The generator is split into two independent parts that run separately.
Part 1 produces the structured CSV data. Part 2 produces the unstructured
HTML documents and their embeddings. Part 2 depends on Part 1 (it reads
the generated CSVs to reference real entity names and IDs) but Part 1
has no dependency on Part 2.

### Part 1: CSV Data Generator

**What it produces:** All seven CSV files in `data/csv/`.

**Scope:** Pydantic models, Faker-based generators, seeded randomness,
configurable counts via Pydantic Settings. Generates entities in dependency
order (banks and companies first, then customers, then accounts, then
stocks, holdings, and transactions). Writes CSV files to the output
directory.

**Entry point:** `python -m data.csv_generator` (reads config from
environment variables, writes CSVs).

**No external dependencies beyond Faker and Pydantic.** Runs locally
without Databricks access.

#### Part 1: Implementation Status

- [x] `data/csv_generator/config.py` -- Pydantic Settings with all configurable counts and distributions
- [x] `data/csv_generator/models.py` -- Pydantic models matching exact CSV schemas (Customer, Account, Bank, Transaction, Company, Stock, PortfolioHolding)
- [x] `data/csv_generator/generators.py` -- Generator functions for all 7 entity types with seeded randomness
- [x] `data/csv_generator/main.py` -- Orchestrator: dependency-ordered generation, CSV writing, summary logging
- [x] `data/csv_generator/__main__.py` -- Supports `python -m data.csv_generator`
- [x] `pyproject.toml` updated with `[generator]` optional dependency group (faker, pydantic-settings)
- [x] All 7 CSV files regenerated in `data/csv/` with correct schemas and ID formats
- [x] Reproducibility verified (seed=42 produces identical output on re-run)

**Generated counts (2026-04-08):**

| Entity | Count | Notes |
|--------|-------|-------|
| Customers | 500 | 150 labeled (50 per risk class), 350 unlabeled |
| Banks | 50 | |
| Companies | 200 | |
| Stocks | 200 | One per company |
| Accounts | 914 | 317 Checking, 302 Savings, 295 Investment |
| Portfolio Holdings | 882 | Only on investment accounts |
| Transactions | 3,687 | 3-5 per account |

### Part 2: HTML Documents and Embeddings

**What it produces:** HTML files in `data/html/` and the embedding JSON
file in `data/embeddings/`.

**Scope:** Reads the CSVs from Part 1 to pick real customer names, company
names, bank names, and financial details. Uses an LLM endpoint to generate
realistic narrative HTML documents (customer profiles, company analyses,
sector reports, investment guides). Then chunks the HTML and calls the
Databricks embedding endpoint to produce real 1024-dim vectors.

**Entry points:**
- Local: `python -m data.html_generator` (live) or `--dry-run` (no endpoints)
- Databricks: `uv run python -m cli submit generate_html.py` (runs on-cluster)

**Requires Databricks access** for both the LLM endpoint (document
generation) and the embedding endpoint (vector generation). Can be run
after Part 1 is complete. The `html_generator` package is included in the
wheel so it's importable on-cluster.

#### Part 2: Implementation Status

- [x] `data/html_generator/__init__.py` -- Package init
- [x] `data/html_generator/config.py` -- Pydantic Settings for LLM endpoint, embedding endpoint, document counts, chunk size, output paths
- [x] `data/html_generator/models.py` -- Pydantic models for DocumentRecord, ChunkRecord, EmbeddingOutput matching exact `document_chunks_embedded.json` schema
- [x] `data/html_generator/document_generator.py` -- CSV reader, entity selectors, LLM prompt templates for all 6 document types, dry-run HTML templates, LLM caller
- [x] `data/html_generator/embedding_generator.py` -- HTML-to-text stripping, 4000-char/200-overlap chunking, batch embedding via Databricks endpoint, dry-run random vectors, full JSON output assembly
- [x] `data/html_generator/main.py` -- CLI orchestrator with `--dry-run`, `--html-only`, `--seed` flags
- [x] `data/html_generator/__main__.py` -- Supports `python -m data.html_generator`
- [x] `agent_modules/generate_html.py` -- Databricks job entry point (reads CSVs from volume, writes results back)
- [x] `html_generator` included in wheel via `pyproject.toml` build config
- [x] Dry-run tested: 41 documents, 41 chunks (dry-run templates are shorter; LLM output will produce ~60 chunks at full length)
- [ ] Live run with Databricks LLM endpoint (requires `DATABRICKS_HOST` + `DATABRICKS_TOKEN`)
- [ ] Live run with Databricks embedding endpoint (real 1024-dim vectors)
- [ ] Final output written to `data/html/` and `data/embeddings/`

**Document type breakdown (configured defaults):**

| Type | Count |
|------|-------|
| Customer profiles | 8 |
| Company analyses | 12 |
| Sector/market analyses | 7 |
| Investment/planning guides | 7 |
| Bank profiles | 4 |
| Regulatory/compliance docs | 3 |
| **Total** | **41** |