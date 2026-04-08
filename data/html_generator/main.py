"""Orchestrator for HTML document and embedding generation.

Usage:
    # Dry-run mode (no Databricks endpoints needed)
    python -m data.html_generator --dry-run

    # Live mode (auth via databricks-sdk — workspace client or env vars)
    python -m data.html_generator

    # Override document counts
    GEN_NUM_CUSTOMER_PROFILES=10 python -m data.html_generator --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from dotenv import load_dotenv

from .config import GeneratorConfig
from .document_generator import generate_documents
from .embedding_generator import (
    build_embedding_output,
    write_embedding_json,
    write_html_files,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate HTML documents and embedding vectors "
        "for graph-feature-forge.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Use templates and random vectors instead of LLM/embedding endpoints.",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        default=False,
        help="Generate HTML documents but skip embedding generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides GEN_SEED env var).",
    )
    args = parser.parse_args(argv)

    # Load .env so DATABRICKS_PROFILE is available to the SDK.
    # Map DATABRICKS_PROFILE -> DATABRICKS_CONFIG_PROFILE (the var the
    # SDK recognises) so that DatabricksEmbeddings and other SDK code
    # that doesn't accept an explicit profile argument also picks it up.
    load_dotenv()
    profile = os.environ.get("DATABRICKS_PROFILE")
    if profile and not os.environ.get("DATABRICKS_CONFIG_PROFILE"):
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile

    # Build config from env, then apply CLI overrides
    cfg = GeneratorConfig()

    if args.dry_run:
        cfg.dry_run = True

    if args.seed is not None:
        cfg.seed = args.seed

    # For live mode, verify Databricks SDK can authenticate
    if not cfg.dry_run:
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient(profile=profile)
            _ = w.config.host  # Trigger auth validation
        except Exception as exc:
            print(
                f"ERROR: Live mode requires Databricks authentication. "
                f"Set DATABRICKS_PROFILE in .env (pointing to a "
                f"~/.databrickscfg profile), or set DATABRICKS_HOST + "
                f"DATABRICKS_TOKEN.  Use --dry-run for local testing.\n"
                f"  Detail: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Validate CSV directory exists
    if not cfg.csv_dir.exists():
        print(
            f"WARNING: CSV directory {cfg.csv_dir} does not exist.  "
            "Run Part 1 (data_generator) first to generate CSVs.  "
            "Continuing with empty entity data...",
            file=sys.stderr,
        )

    print("=" * 60)
    print("HTML Document & Embedding Generator")
    print("=" * 60)
    print(f"Mode:            {'DRY RUN' if cfg.dry_run else 'LIVE'}")
    print(f"Seed:            {cfg.seed}")
    print(f"CSV source:      {cfg.csv_dir}")
    print(f"HTML output:     {cfg.html_output_dir}")
    print(f"Embedding output:{cfg.embedding_output_dir}")
    print(f"Max workers:     {cfg.max_workers}")
    print(f"Target docs:     {cfg.total_documents}")
    print(f"  Customer:      {cfg.num_customer_profiles}")
    print(f"  Company:       {cfg.num_company_analyses}")
    print(f"  Sector:        {cfg.num_sector_analyses}")
    print(f"  Guides:        {cfg.num_investment_guides}")
    print(f"  Banks:         {cfg.num_bank_profiles}")
    print(f"  Regulatory:    {cfg.num_regulatory_docs}")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Generate HTML documents
    print("\n[1/3] Generating HTML documents...")
    documents = generate_documents(cfg)

    # Step 2: Write HTML files
    print("\n[2/3] Writing HTML files...")
    write_html_files(documents, cfg.html_output_dir)

    if args.html_only:
        elapsed = time.time() - t0
        print(f"\nDone (HTML only) in {elapsed:.1f}s")
        return

    # Step 3: Chunk and embed
    print("\n[3/3] Chunking and embedding...")
    embedding_output = build_embedding_output(documents, cfg)
    write_embedding_json(embedding_output, cfg.embedding_output_dir)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  {len(documents)} HTML documents")
    print(f"  {embedding_output.metadata.chunk_count} embedding chunks")


if __name__ == "__main__":
    main()
