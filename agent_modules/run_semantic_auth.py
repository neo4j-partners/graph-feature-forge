"""Incremental enrichment pipeline: extract -> synthesis -> analyze -> dedup -> dual-write.

Each cycle:

1. Ensure base Delta tables exist (CSV load, idempotent)
2. Ensure enrichment_log table exists
3. Extract enrichment-only data from Neo4j (skip base labels/types)
4. Load structured data + prior enrichment context
5. Synthesize gap analysis via LLM
6. Run four DSPy analyzers for schema-level suggestions
7. Resolve schema-level suggestions into instance proposals
8. Deduplicate against enrichment_log
9. Filter instance proposals by confidence
10. Dual-write HIGH proposals to Neo4j + enrichment_log

Usage:
    python -m cli upload --wheel
    python -m cli upload run_semantic_auth.py
    python -m cli submit run_semantic_auth.py
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Pipeline configuration from environment variables and CLI flags."""

    source_catalog: str
    source_schema: str
    llm_endpoint: str
    embedding_endpoint: str
    neo4j_uri: str | None
    neo4j_username: str | None
    neo4j_password: str | None
    neo4j_database: str
    enable_tracing: bool
    execute: bool

    @classmethod
    def from_env(cls) -> PipelineConfig:
        """Load config from environment variables and CLI flags."""
        from semantic_auth import inject_params

        inject_params()

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--execute", action="store_true")
        parser.add_argument("--enable-tracing", action="store_true")
        flags, _ = parser.parse_known_args()

        return cls(
            source_catalog=os.getenv("SOURCE_CATALOG", "neo4j_augmentation_demo"),
            source_schema=os.getenv("SOURCE_SCHEMA", "raw_data"),
            llm_endpoint=os.getenv("LLM_ENDPOINT", "databricks-claude-sonnet-4-6"),
            embedding_endpoint=os.getenv(
                "EMBEDDING_ENDPOINT", "databricks-gte-large-en"
            ),
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_username=os.getenv("NEO4J_USERNAME"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            enable_tracing=flags.enable_tracing or os.getenv("ENABLE_TRACING", "").lower() == "true",
            execute=flags.execute or os.getenv("EXECUTE", "").lower() == "true",
        )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _authenticate() -> tuple[Any, str, str]:
    """Connect to Databricks and return (client, host, token).

    On serverless runtimes ``config.token`` is None, so the token is
    extracted from the auth headers as a fallback.  The host and token
    are also exported to ``os.environ`` for LiteLLM auto-detection.
    """
    from databricks.sdk import WorkspaceClient

    wc = WorkspaceClient()
    host = wc.config.host
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

    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token

    return wc, host, token


def _base_tables_exist(execute_sql: Any, cfg: PipelineConfig) -> bool:
    """Check whether the 14 base Delta tables already exist."""
    from semantic_auth.graph_schema import NODE_TABLE_NAMES, RELATIONSHIP_TABLE_NAMES

    expected = set(NODE_TABLE_NAMES + RELATIONSHIP_TABLE_NAMES)
    try:
        rows = execute_sql(
            f"SHOW TABLES IN `{cfg.source_catalog}`.`{cfg.source_schema}`"
        )
        existing = {r.get("tableName", r.get("table_name", "")) for r in rows}
        return expected.issubset(existing)
    except Exception:
        return False


def _ensure_base_tables(execute_sql: Any, cfg: PipelineConfig) -> None:
    """Create base Delta tables from CSV if they don't exist."""
    if _base_tables_exist(execute_sql, cfg):
        print("  Base tables already exist — skipping CSV load")
        return

    from semantic_auth.loading import load_all

    volume_path = os.getenv("DATABRICKS_VOLUME_PATH", "")
    if not volume_path:
        catalog = os.getenv("CATALOG_NAME", cfg.source_catalog)
        schema_name = os.getenv("SCHEMA_NAME", cfg.source_schema)
        volume_name = os.getenv("VOLUME_NAME", "pipeline_data")
        volume_path = f"/Volumes/{catalog}/{schema_name}/{volume_name}"

    print("  Loading base tables from CSV ...")
    load_all(execute_sql, cfg.source_catalog, cfg.source_schema, volume_path)


def _extract_enrichment_data(cfg: PipelineConfig) -> None:
    """Extract enrichment-only data from Neo4j (skip base labels/types)."""
    from semantic_auth.extraction import extract_graph
    from semantic_auth.graph_schema import BASE_NODE_LABELS, BASE_RELATIONSHIP_TYPES

    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
    except Exception:
        print("  WARNING: No Spark session — skipping Neo4j extraction")
        return

    t0 = time.time()
    extract_graph(
        spark=spark,
        uri=cfg.neo4j_uri,
        username=cfg.neo4j_username,
        password=cfg.neo4j_password,
        database=cfg.neo4j_database,
        catalog=cfg.source_catalog,
        schema=cfg.source_schema,
        base_node_labels=BASE_NODE_LABELS,
        base_rel_types=BASE_RELATIONSHIP_TYPES,
    )
    elapsed = time.time() - t0
    print(f"  Enrichment extraction complete in {elapsed:.1f}s")


def _configure_dspy(
    host: str, token: str, llm_endpoint: str, enable_tracing: bool
) -> None:
    """Configure DSPy LM with validation and provider fallback."""
    if enable_tracing:
        import warnings

        import mlflow

        # Suppress Pydantic serialization warnings from MLflow autolog
        # when it serializes LiteLLM response objects (Message/Choices
        # field count mismatches).
        warnings.filterwarnings(
            "ignore",
            message=".*PydanticSerializationUnexpectedValue.*",
            category=UserWarning,
        )

        experiment_path = os.getenv("DATABRICKS_WORKSPACE_DIR", "/Shared/semantic-auth")
        mlflow.set_experiment(experiment_path)
        mlflow.dspy.autolog()
        print(f"  MLflow DSPy tracing enabled (experiment: {experiment_path})")

    import dspy

    lm = dspy.LM(
        f"databricks/{llm_endpoint}",
        api_key=token,
        api_base=f"{host}/serving-endpoints",
        temperature=0.1,
        max_tokens=30000,
    )

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


def _run_synthesis(
    data: Any,
    retrieval: Any,
    llm_endpoint: str,
    enrichment_context: str | None = None,
) -> str:
    """Synthesize gap analysis from structured data and retrieved documents."""
    from semantic_auth.synthesis import fetch_gap_analysis, make_sdk_caller

    print("\nStep 3/5: Running gap analysis synthesis ...")
    t0 = time.time()
    llm_caller = make_sdk_caller(endpoint=llm_endpoint, max_tokens=8192)
    gap_analysis = fetch_gap_analysis(
        data, retrieval, llm_caller,
        enrichment_context=enrichment_context,
    )
    elapsed = time.time() - t0
    print(f"  Synthesis complete: {len(gap_analysis):,} chars in {elapsed:.1f}s")
    print(f"  Preview: {gap_analysis[:200]}...")
    return gap_analysis


def _run_analyzers(gap_analysis: str) -> Any:
    """Run four concurrent DSPy analyzers for schema-level suggestions."""
    from semantic_auth.analyzers import GraphAugmentationAnalyzer
    from semantic_auth.reporting import print_response_summary

    print("\nStep 4/5: Running DSPy analyzers (4 concurrent) ...")
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

    return response


def _resolve_proposals(response: Any, gap_analysis: str) -> Any | None:
    """Resolve schema suggestions to instance proposals and filter by confidence.

    Returns a :class:`FilteredProposals` instance, or ``None`` if no
    instance proposals were generated.
    """
    from semantic_auth.analyzers import InstanceResolver
    from semantic_auth.reporting import print_filtered_proposals
    from semantic_auth.schemas import FilteredProposals

    print("\nStep 5/5: Resolving to instance proposals ...")
    t0 = time.time()
    resolver = InstanceResolver()
    resolution = resolver(response=response, document_context=gap_analysis)
    elapsed = time.time() - t0
    print(f"  Resolution complete in {elapsed:.1f}s")

    if not resolution.proposals:
        return None

    filtered = FilteredProposals.from_proposals(resolution.proposals)
    print_filtered_proposals(filtered)
    return filtered


def _ensure_results_dir(wc: Any) -> str | None:
    """Create the volume results directory if configured, return path or None."""
    volume_path = os.getenv("DATABRICKS_VOLUME_PATH", "")
    if not volume_path:
        return None

    results_dir = f"{volume_path}/results"
    if not results_dir.startswith("/Volumes"):
        results_dir = f"/Volumes{results_dir}"

    try:
        wc.files.create_directory(results_dir)
    except Exception:
        pass  # directory may already exist

    return results_dir


def _save_to_volume(wc: Any, results_dir: str, filename: str, content: str) -> None:
    """Upload JSON content to a UC volume results directory."""
    path = f"{results_dir}/{filename}"
    wc.files.upload(
        file_path=path,
        contents=io.BytesIO(content.encode()),
        overwrite=True,
    )
    print(f"  Saved: {path}")


def _write_back(
    cfg: PipelineConfig,
    proposals: list,
    run_id: str,
    enrichment_store: Any = None,
) -> None:
    """Dual-write HIGH-confidence proposals to Neo4j + enrichment log."""
    from semantic_auth.writeback import Neo4jWriter

    dry_run = not cfg.execute

    with Neo4jWriter(
        cfg.neo4j_uri, cfg.neo4j_username, cfg.neo4j_password, cfg.neo4j_database,
        enrichment_store=enrichment_store if not dry_run else None,
    ) as writer:
        writer.write_proposals(proposals, run_id=run_id, dry_run=dry_run)

    if dry_run:
        print("\n  Pass --execute to write these proposals to Neo4j.")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = PipelineConfig.from_env()

    if not all([cfg.neo4j_uri, cfg.neo4j_username, cfg.neo4j_password]):
        print("ERROR: Neo4j connection required.")
        print("  Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env")
        sys.exit(1)

    run_id = uuid.uuid4().hex[:12]
    print("=" * 60)
    print(f"Semantic Auth Pipeline  (run {run_id})")
    print("=" * 60)

    wc, host, token = _authenticate()

    # --- Step 1: Ensure base tables + enrichment log exist ---------------
    from semantic_auth.enrichment_store import EnrichmentStore
    from semantic_auth.structured_data import StructuredDataAccess, make_spark_executor

    execute_sql = make_spark_executor()

    print("\nStep 1/5: Ensuring base tables and enrichment log ...")
    enrichment_store = EnrichmentStore(
        execute_sql, cfg.source_catalog, cfg.source_schema,
    )
    enrichment_store.ensure_table()
    print("  Enrichment log table: ready")

    _ensure_base_tables(execute_sql, cfg)

    # --- Step 2: Extract enrichment data from Neo4j ----------------------
    print("\nStep 2/5: Extracting enrichment data from Neo4j ...")
    _extract_enrichment_data(cfg)

    # --- Load enrichment context for synthesis ---------------------------
    enrichment_context = enrichment_store.format_context()
    prior_count = enrichment_store.count()
    if prior_count:
        print(f"  Prior enrichments: {prior_count} proposals from previous runs")

    # --- Configure DSPy --------------------------------------------------
    _configure_dspy(host, token, cfg.llm_endpoint, cfg.enable_tracing)

    # Structured data access
    data = StructuredDataAccess(
        execute_sql=execute_sql,
        catalog=cfg.source_catalog,
        schema=cfg.source_schema,
    )
    print("  Structured data: Spark executor")

    # Document retrieval (Neo4j vector index)
    import neo4j as neo4j_lib
    from semantic_auth.retrieval import Neo4jRetrieval, make_sdk_embedder

    neo4j_driver = neo4j_lib.GraphDatabase.driver(
        cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password)
    )

    try:
        retrieval = Neo4jRetrieval(
            driver=neo4j_driver,
            embedder=make_sdk_embedder(cfg.embedding_endpoint),
            database=cfg.neo4j_database,
        )
        print("  Document retrieval: Neo4j vector index (graph-aware)")

        gap_analysis = _run_synthesis(
            data, retrieval, cfg.llm_endpoint,
            enrichment_context=enrichment_context or None,
        )
        response = _run_analyzers(gap_analysis)

        results_dir = _ensure_results_dir(wc)
        if results_dir:
            _save_to_volume(
                wc,
                results_dir,
                f"enrichment_results_{run_id}.json",
                response.model_dump_json(indent=2),
            )

        filtered = _resolve_proposals(response, gap_analysis)
        if not filtered:
            print("\n  No instance proposals generated. Pipeline complete.")
            return

        if results_dir:
            _save_to_volume(
                wc,
                results_dir,
                f"instance_proposals_{run_id}.json",
                filtered.model_dump_json(indent=2),
            )

        if not filtered.auto_approve:
            print("\n  No HIGH-confidence proposals to write. Pipeline complete.")
            return

        # --- Deduplicate against enrichment log --------------------------
        before_dedup = len(filtered.auto_approve)
        deduped = enrichment_store.deduplicate(filtered.auto_approve)
        if before_dedup != len(deduped):
            print(
                f"\n  Dedup: {before_dedup - len(deduped)} duplicate proposals "
                f"removed ({len(deduped)} remaining)"
            )

        if not deduped:
            print("\n  All proposals already exist. Pipeline complete.")
            return

        _write_back(cfg, deduped, run_id, enrichment_store=enrichment_store)
        print(f"\n  Pipeline complete (run {run_id})")

    finally:
        neo4j_driver.close()


if __name__ == "__main__":
    main()
