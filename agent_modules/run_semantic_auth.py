"""Full pipeline: synthesis -> DSPy analyzers -> structured output.

Replaces the Supervisor Agent-backed pipeline from the workshop (Lab 7)
with direct LLM calls against ``databricks-claude-sonnet-4-6``.

The runner auto-attaches the ``semantic_auth`` wheel when this script is
submitted (matches ``wheel_package="semantic_auth"`` in ``cli/__init__.py``).

Usage:
    python -m cli upload --wheel
    python -m cli upload run_semantic_auth.py
    python -m cli submit run_semantic_auth.py
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Auth Pipeline")
    parser.add_argument("--source-catalog", default="neo4j_augmentation_demo")
    parser.add_argument("--source-schema", default="raw_data")
    parser.add_argument("--catalog-name", default="semantic-auth")
    parser.add_argument("--schema-name", default="enrichment")
    parser.add_argument("--volume-name", default="source-data")
    parser.add_argument("--llm-endpoint", default="databricks-claude-sonnet-4-6")
    parser.add_argument("--embedding-endpoint", default="databricks-gte-large-en")
    parser.add_argument("--warehouse-id", default=None)
    parser.add_argument("--enable-tracing", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Semantic Auth Pipeline")
    print("=" * 60)

    # -- Step 1: Databricks auth -------------------------------------------

    from databricks.sdk import WorkspaceClient

    wc = WorkspaceClient()
    host = wc.config.host
    # On serverless runtime, config.token is None — extract from auth headers
    token = wc.config.token
    if not token:
        try:
            headers = wc.config.authenticate()
            token = headers.get("Authorization", "").removeprefix("Bearer ")
        except Exception:
            token = None
    print(f"  Connected to {host}")

    if not token:
        print("ERROR: Could not obtain Databricks API token.")
        print("  Set DATABRICKS_HOST + DATABRICKS_TOKEN, or configure")
        print("  a ~/.databrickscfg profile via DATABRICKS_PROFILE.")
        sys.exit(1)

    # Export for LiteLLM's Databricks provider auto-detection
    import os
    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token

    # -- Step 2: MLflow tracing (optional) ---------------------------------

    if args.enable_tracing:
        import mlflow

        mlflow.dspy.autolog()
        print("  MLflow DSPy tracing enabled")

    # -- Step 3: Configure DSPy with standard LM ---------------------------
    #
    # Uses the LiteLLM ``databricks/`` provider which speaks the
    # OpenAI-compatible chat completions API that Databricks foundation
    # model endpoints expose.  No custom BaseLM adapter needed.

    import dspy

    lm = dspy.LM(
        f"databricks/{args.llm_endpoint}",
        api_key=token,
        api_base=f"{host}/serving-endpoints",
        temperature=0.1,
        max_tokens=4000,
    )

    # Quick validation before running the full pipeline
    try:
        test_result = lm("Say hello")
        print(f"  DSPy configured: {args.llm_endpoint} (validated)")
    except Exception as exc:
        print(f"  DSPy LM validation failed: {type(exc).__name__}: {exc}")
        print(f"  Host: {host}")
        print(f"  Trying openai/ provider fallback...")
        lm = dspy.LM(
            f"openai/{args.llm_endpoint}",
            api_key=token,
            api_base=f"{host}/serving-endpoints",
            temperature=0.1,
            max_tokens=4000,
        )
        try:
            test_result = lm("Say hello")
            print(f"  DSPy configured: openai/{args.llm_endpoint} (validated)")
        except Exception as exc2:
            print(f"  openai/ provider also failed: {type(exc2).__name__}: {exc2}")
            sys.exit(1)

    dspy.configure(lm=lm)

    # -- Step 4: Structured data access ------------------------------------

    from semantic_auth.structured_data import (
        StructuredDataAccess,
        make_sdk_executor,
        make_spark_executor,
    )

    # Prefer Spark when available (serverless/cluster) — avoids warehouse cold start.
    # Fall back to SDK executor with warehouse_id for local development.
    try:
        executor = make_spark_executor()
        print("  Structured data: Spark executor")
    except RuntimeError:
        if args.warehouse_id:
            executor = make_sdk_executor(args.warehouse_id)
            print(f"  Structured data: SDK executor (warehouse {args.warehouse_id})")
        else:
            print("ERROR: No Spark session available and no WAREHOUSE_ID provided.")
            print("  Provide WAREHOUSE_ID in .env for local execution,")
            print("  or run on a cluster with Spark available.")
            sys.exit(1)

    data = StructuredDataAccess(
        execute_sql=executor,
        catalog=args.source_catalog,
        schema=args.source_schema,
    )

    # -- Step 5: Document retrieval ----------------------------------------

    from semantic_auth.retrieval import DocumentRetrieval, make_sdk_embedder

    embeddings_path = (
        f"/Volumes/{args.catalog_name}/{args.schema_name}"
        f"/{args.volume_name}/embeddings/document_chunks_embedded.json"
    )
    embedder = make_sdk_embedder(args.embedding_endpoint)
    retrieval = DocumentRetrieval.from_json_path(embeddings_path, embedder)
    print(f"  Document retrieval: loaded from {embeddings_path}")

    # -- Step 6: Synthesis (gap analysis) ----------------------------------

    from semantic_auth.synthesis import fetch_gap_analysis, make_sdk_caller

    print("\nStep 1/2: Running gap analysis synthesis ...")
    t0 = time.time()
    llm_caller = make_sdk_caller(
        endpoint=args.llm_endpoint,
        max_tokens=8192,
    )
    gap_analysis = fetch_gap_analysis(data, retrieval, llm_caller)
    elapsed = time.time() - t0
    print(f"  Synthesis complete: {len(gap_analysis):,} chars in {elapsed:.1f}s")
    print(f"  Preview: {gap_analysis[:200]}...")

    # -- Step 7: DSPy analyzers --------------------------------------------

    from semantic_auth.analyzers import GraphAugmentationAnalyzer
    from semantic_auth.reporting import print_response_summary

    print("\nStep 2/2: Running DSPy analyzers (4 concurrent) ...")
    t0 = time.time()
    analyzer = GraphAugmentationAnalyzer()
    response = analyzer(document_context=gap_analysis)
    elapsed = time.time() - t0

    # -- Step 8: Results ---------------------------------------------------

    print_response_summary(response)

    print(f"\n  Pipeline complete in {elapsed:.1f}s (analyzer phase)")
    print(f"  Total suggestions: {response.total_suggestions}")
    print(f"  High confidence:   {response.high_confidence_count}")

    if not response.success:
        print("\n  WARNING: Some analyses failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
