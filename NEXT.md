# Current Status and Next Steps

## Pipeline Status (2026-04-06)

The full graph enrichment pipeline runs end-to-end on Databricks serverless compute. A single `python -m cli submit run_semantic_auth.py` builds, uploads, and executes the pipeline without any manual UI configuration in the Databricks workspace.

### Successful Run (run ID 649594017312543)

- **Synthesis**: 21,081 chars in 122s via `databricks-claude-sonnet-4-6`
- **DSPy Analyzers**: 4 concurrent analyses in 71s
- **Output**: 17 suggestions, 13 high confidence
  - 10 Investment Themes (AI in Financial Services, Digital Payments, Robo-Advisory, etc.)
  - 3 Suggested Nodes (FINANCIAL_GOAL, INVESTMENT_INTEREST, INVESTMENT_THEME)
  - 2 Suggested Relationships (INTERESTED_IN, HAS_GOAL)
  - 12 Suggested Attributes (occupation, employer, investment_philosophy, life_stage, etc.)

### Validation Against Known Gap Analysis Results

The pipeline correctly identifies the three key findings from the workshop's Supervisor Agent pipeline:
- James Anderson: 100% Technology holdings vs. stated renewable energy interest
- Maria Rodriguez: ESG/sustainable investing interest not reflected in portfolio
- Robert Chen: aggressive growth profile identified

Suggestion categories match: new node types (FINANCIAL_GOAL, INVESTMENT_INTEREST), relationship types (INTERESTED_IN, HAS_GOAL), and attributes (occupation, investment_philosophy) all align with the workshop's Lab 7 output.

## Issues Fixed During Integration

1. **`databricks-job-runner` not on PyPI**: Moved from core dependencies to `[cli]` extra so the wheel installs cleanly on serverless.
2. **DSPy/Pydantic not in wheel**: Moved from `[enrichment]` extra to core dependencies since the wheel's modules import them at module level.
3. **Serverless auth token**: `wc.config.token` returns `None` on serverless runtime. Fixed by extracting token from `wc.config.authenticate()` headers.
4. **Spark vs SDK executor**: Pipeline now prefers `make_spark_executor()` when PySpark is available (serverless/cluster), falling back to `make_sdk_executor()` only for local development. Avoids SQL warehouse cold start delays.
5. **SDK HTTP timeout**: Increased to 600s for the synthesis LLM call (comprehensive gap analysis takes ~120s).
6. **LiteLLM endpoint routing**: LiteLLM's `databricks/` provider needed `api_base` set to `{host}/serving-endpoints` (not just `{host}`) plus `DATABRICKS_HOST`/`DATABRICKS_TOKEN` exported as env vars for auto-detection.
7. **`build_params` signature**: Updated to accept `(config, script)` to match current `databricks-job-runner` API.

## What to Test and Improve

### Truncation Warnings (Quick Fix)

Three of four analyzers hit the `max_tokens=4000` limit and produced truncated output. The pipeline still succeeded because DSPy's `ChatAdapter` extracted valid JSON before the truncation point, but some suggestions may have been lost. Increase to `max_tokens=8000` in the `dspy.LM` configuration in `run_semantic_auth.py`.

### MLflow Tracing (Phase 4 Remaining)

The `--enable-tracing` flag exists but hasn't been tested. Run:
```bash
python -m cli submit run_semantic_auth.py --extra-args "--enable-tracing"
```
Verify that MLflow traces appear in the experiment UI showing: synthesis call timing, four parallel analyzer calls, and structured output extraction. This requires `mlflow[databricks]` to be installable on the serverless runtime.

### Retrieval Quality Tuning

The in-memory cosine similarity retrieval returns reasonable results (scores 0.55-0.67) but could be improved:
- **Re-chunking**: The pre-computed chunks range from 77 to 4,094 chars. More uniform chunk sizes with overlap could improve retrieval precision.
- **Query-specific top-k**: Some query types (investment themes) benefit from more document context, while others (data quality) need fewer but more targeted chunks. The current `top_k=5` is uniform across all query types.
- **Hybrid retrieval**: Combine embedding similarity with keyword matching for better recall on specific entity names (e.g., "James Anderson").

### Synthesis Prompt Optimization

The comprehensive prompt sends all structured data + all retrieved docs in a single call. Alternatives:
- **Per-query-type synthesis**: Run 4 separate synthesis calls (one per gap analysis type) instead of one comprehensive call. More tokens total but each prompt is more focused. The methods already exist (`analyze_interest_holding_gaps()`, etc.).
- **Context windowing**: Only include structured data relevant to each query type rather than all three queries every time.

### Output Persistence

Currently the pipeline prints results to stdout. Future additions:
- Write `AugmentationResponse` as JSON to a Delta table in the `semantic-auth.enrichment` schema.
- Write back to Neo4j as new nodes/relationships (the original goal from start.md Phase 5+).
- Store synthesis text alongside structured output for audit/debugging.

### Confidence Calibration

13 of 17 suggestions are "high confidence." This may indicate the confidence thresholds in the DSPy signatures are too generous. Compare against a human-reviewed ground truth set to calibrate whether "high" confidence suggestions are actually actionable.

### Cost Tracking

Each pipeline run makes ~5 LLM calls to `databricks-claude-sonnet-4-6` (1 synthesis + 4 analyzer). Track token usage and cost per run via the Databricks system tables (`system.billing.usage`) or MLflow tracing metadata.
