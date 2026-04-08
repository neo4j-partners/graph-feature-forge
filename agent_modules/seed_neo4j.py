"""Seed a Neo4j instance from Delta tables and embeddings JSON.

Replicates the graph-enrichment workshop's Lab 1 import:
- 7 node types (Customer, Bank, Account, Company, Stock, Position, Transaction)
- 7 relationship types (HAS_ACCOUNT, AT_BANK, OF_COMPANY, etc.)
- Document graph (Documents, Chunks with embeddings, DESCRIBES/FROM_DOCUMENT/NEXT_CHUNK)
- Vector and full-text indexes

Usage:
    python -m cli upload --wheel
    python -m cli upload seed_neo4j.py
    python -m cli submit seed_neo4j.py
"""

from __future__ import annotations

import os
import sys
import time


def main() -> None:
    from graph_feature_forge import inject_params

    inject_params()

    source_catalog = os.getenv("SOURCE_CATALOG", "neo4j_augmentation_demo")
    source_schema = os.getenv("SOURCE_SCHEMA", "raw_data")
    catalog_name = os.getenv("CATALOG_NAME", "graph-feature-forge")
    schema_name = os.getenv("SCHEMA_NAME", "enrichment")
    volume_name = os.getenv("VOLUME_NAME", "source-data")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    print("=" * 60)
    print("Seed Neo4j from Delta Tables")
    print("=" * 60)

    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        print("ERROR: Neo4j connection required.")
        print("  Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env")
        sys.exit(1)

    print(f"  Neo4j: {neo4j_uri}")
    print(f"  Database: {neo4j_database}")

    from graph_feature_forge.data.structured_data import make_spark_executor

    executor = make_spark_executor()
    print("  Data source: Spark executor")

    embeddings_path = (
        f"/Volumes/{catalog_name}/{schema_name}"
        f"/{volume_name}/embeddings/document_chunks_embedded.json"
    )
    print(f"  Embeddings: {embeddings_path}")

    from graph_feature_forge.graph.seeding import seed_neo4j

    print(f"\n  Source: {source_catalog}.{source_schema}")

    t0 = time.time()
    seed_neo4j(
        execute_sql=executor,
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        source_catalog=source_catalog,
        source_schema=source_schema,
        embeddings_path=embeddings_path,
    )
    elapsed = time.time() - t0

    print(f"\n  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
