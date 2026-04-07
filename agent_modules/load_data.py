"""Load raw data into Delta tables from CSV files on a UC volume.

Uploads CSV data to Delta tables matching the Neo4j Spark Connector
export format, so seed_neo4j.py can read them directly.

Usage:
    python -m cli upload --data
    python -m cli upload --wheel
    python -m cli upload load_data.py
    python -m cli submit load_data.py
"""

from __future__ import annotations

import os
import sys
import time


def main() -> None:
    from semantic_auth import inject_params

    inject_params()

    source_catalog = os.getenv("SOURCE_CATALOG", "neo4j_augmentation_demo")
    source_schema = os.getenv("SOURCE_SCHEMA", "raw_data")
    catalog_name = os.getenv("CATALOG_NAME", "semantic-auth")
    schema_name = os.getenv("SCHEMA_NAME", "enrichment")
    volume_name = os.getenv("VOLUME_NAME", "source-data")

    volume_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"

    print("=" * 60)
    print("Load Raw Data into Delta Tables")
    print("=" * 60)
    print(f"  Volume:  {volume_path}")
    print(f"  Target:  {source_catalog}.{source_schema}")

    # -- SQL executor ----------------------------------------------------------

    from semantic_auth.structured_data import make_spark_executor

    try:
        executor = make_spark_executor()
        print("  Executor: Spark")
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        print("  This script must run on a Databricks cluster or serverless.")
        sys.exit(1)

    # -- Load ------------------------------------------------------------------

    from semantic_auth.loading import load_all

    t0 = time.time()
    counts = load_all(
        execute_sql=executor,
        catalog=source_catalog,
        schema=source_schema,
        volume_path=volume_path,
    )
    elapsed = time.time() - t0

    total = sum(counts.values())
    print(f"\n  Created {len(counts)} tables, {total} total rows in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
