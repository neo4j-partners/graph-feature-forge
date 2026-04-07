"""Full pipeline: extract -> synthesis -> analyzers -> resolve -> filter -> write-back.

Implements the enrichment loop described in graph_enrichment_v3.md:

1. Extract graph state from Neo4j to Delta tables (Spark Connector)
2. Load structured data from Delta tables
3. Retrieve relevant documents via embeddings
4. Synthesize gap analysis via LLM
5. Run four DSPy analyzers for schema-level suggestions
6. Resolve schema-level suggestions into instance proposals
7. Filter instance proposals by confidence
8. Write HIGH-confidence proposals back to Neo4j

Usage:
    python -m cli upload --wheel
    python -m cli upload run_semantic_auth.py
    python -m cli submit run_semantic_auth.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid


def main() -> None:
    from semantic_auth import inject_params

    inject_params()

    # Config from .env (forwarded by Runner as KEY=VALUE params)
    source_catalog = os.getenv("SOURCE_CATALOG", "neo4j_augmentation_demo")
    source_schema = os.getenv("SOURCE_SCHEMA", "raw_data")
    catalog_name = os.getenv("CATALOG_NAME", "semantic-auth")
    schema_name = os.getenv("SCHEMA_NAME", "enrichment")
    volume_name = os.getenv("VOLUME_NAME", "source-data")
    llm_endpoint = os.getenv("LLM_ENDPOINT", "databricks-claude-sonnet-4-6")
    embedding_endpoint = os.getenv("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    # Behavioral flags (parsed separately from inject_params key=value pairs)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-tracing", action="store_true")
    flags, _ = parser.parse_known_args()

    enable_tracing = flags.enable_tracing
    skip_extraction = flags.skip_extraction or not neo4j_uri
    execute = flags.execute

    run_id = uuid.uuid4().hex[:12]

    print("=" * 60)
    print(f"Semantic Auth Pipeline  (run {run_id})")
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
    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token

    # -- Step 2: Neo4j extraction (Gap 0) ----------------------------------

    if not skip_extraction:
        if not all([neo4j_uri, neo4j_username, neo4j_password]):
            print("ERROR: Neo4j connection required for extraction.")
            print("  Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env,")
            print("  or pass --skip-extraction to use existing Delta tables.")
            sys.exit(1)

        from semantic_auth.extraction import extract_graph

        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
        except Exception:
            print("ERROR: Neo4j Spark Connector requires a Spark session.")
            print("  Run on a Databricks cluster with the Neo4j Spark Connector JAR,")
            print("  or pass --skip-extraction to use existing Delta tables.")
            sys.exit(1)

        print("\nStep 1/4: Extracting graph from Neo4j ...")
        t0 = time.time()
        extract_graph(
            spark=spark,
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
            catalog=source_catalog,
            schema=source_schema,
        )
        elapsed = time.time() - t0
        print(f"  Extraction complete in {elapsed:.1f}s")
    else:
        print("\n  Skipping Neo4j extraction (--skip-extraction)")

    # -- Step 3: MLflow tracing (optional) ---------------------------------

    if enable_tracing:
        import mlflow

        mlflow.dspy.autolog()
        print("  MLflow DSPy tracing enabled")

    # -- Step 4: Configure DSPy with standard LM ---------------------------

    import dspy

    lm = dspy.LM(
        f"databricks/{llm_endpoint}",
        api_key=token,
        api_base=f"{host}/serving-endpoints",
        temperature=0.1,
        max_tokens=30000,
    )

    # Quick validation before running the full pipeline
    try:
        lm("Say hello")
        print(f"  DSPy configured: {llm_endpoint} (validated)")
    except Exception as exc:
        print(f"  DSPy LM validation failed: {type(exc).__name__}: {exc}")
        print(f"  Host: {host}")
        print("  Trying openai/ provider fallback...")
        lm = dspy.LM(
            f"openai/{llm_endpoint}",
            api_key=token,
            api_base=f"{host}/serving-endpoints",
            temperature=0.1,
            max_tokens=30000,
        )
        try:
            lm("Say hello")
            print(f"  DSPy configured: openai/{llm_endpoint} (validated)")
        except Exception as exc2:
            print(f"  openai/ provider also failed: {type(exc2).__name__}: {exc2}")
            sys.exit(1)

    dspy.configure(lm=lm)

    # -- Step 5: Structured data access ------------------------------------

    from semantic_auth.structured_data import StructuredDataAccess, make_spark_executor

    executor = make_spark_executor()
    print("  Structured data: Spark executor")

    data = StructuredDataAccess(
        execute_sql=executor,
        catalog=source_catalog,
        schema=source_schema,
    )

    # -- Step 6: Document retrieval ----------------------------------------

    from semantic_auth.retrieval import DocumentRetrieval, make_sdk_embedder

    embeddings_path = (
        f"/Volumes/{catalog_name}/{schema_name}"
        f"/{volume_name}/embeddings/document_chunks_embedded.json"
    )
    embedder = make_sdk_embedder(embedding_endpoint)
    retrieval = DocumentRetrieval.from_json_path(embeddings_path, embedder)
    print(f"  Document retrieval: loaded from {embeddings_path}")

    # -- Step 7: Synthesis (gap analysis) ----------------------------------

    from semantic_auth.synthesis import fetch_gap_analysis, make_sdk_caller

    print("\nStep 2/4: Running gap analysis synthesis ...")
    t0 = time.time()
    llm_caller = make_sdk_caller(
        endpoint=llm_endpoint,
        max_tokens=8192,
    )
    gap_analysis = fetch_gap_analysis(data, retrieval, llm_caller)
    elapsed = time.time() - t0
    print(f"  Synthesis complete: {len(gap_analysis):,} chars in {elapsed:.1f}s")
    print(f"  Preview: {gap_analysis[:200]}...")

    # -- Step 8: DSPy analyzers (schema-level) -----------------------------

    from semantic_auth.analyzers import GraphAugmentationAnalyzer, InstanceResolver
    from semantic_auth.reporting import print_filtered_proposals, print_response_summary
    from semantic_auth.schemas import FilteredProposals

    print("\nStep 3/4: Running DSPy analyzers (4 concurrent) ...")
    t0 = time.time()
    analyzer = GraphAugmentationAnalyzer()
    response = analyzer(document_context=gap_analysis)
    elapsed = time.time() - t0

    print_response_summary(response)

    print(f"\n  Schema-level analysis complete in {elapsed:.1f}s")
    print(f"  Total suggestions: {response.total_suggestions}")
    print(f"  High confidence:   {response.high_confidence_count}")

    if not response.success:
        print("\n  WARNING: Some analyses failed")
        sys.exit(1)

    # -- Step 9: Instance resolution (Gap 1) -------------------------------

    print("\nStep 4/4: Resolving to instance proposals ...")
    t0 = time.time()
    resolver = InstanceResolver()
    resolution = resolver(response=response, document_context=gap_analysis)
    elapsed = time.time() - t0
    print(f"  Resolution complete in {elapsed:.1f}s")

    if not resolution.proposals:
        print("\n  No instance proposals generated. Pipeline complete.")
        return

    # -- Step 10: Confidence filtering (Gap 2) -----------------------------

    filtered = FilteredProposals.from_proposals(resolution.proposals)
    print_filtered_proposals(filtered)

    # -- Step 11: Write-back to Neo4j (Gap 3) ------------------------------

    if not filtered.auto_approve:
        print("\n  No HIGH-confidence proposals to write. Pipeline complete.")
        return

    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        print("\n  No Neo4j connection configured — skipping write-back.")
        print("  Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD to enable.")
        return

    from semantic_auth.writeback import Neo4jWriter

    dry_run = not execute

    with Neo4jWriter(neo4j_uri, neo4j_username, neo4j_password, neo4j_database) as writer:
        writer.write_proposals(filtered.auto_approve, run_id=run_id, dry_run=dry_run)

    if dry_run:
        print("\n  Pass --execute to write these proposals to Neo4j.")

    print(f"\n  Pipeline complete (run {run_id})")


if __name__ == "__main__":
    main()
