"""Generate HTML documents and embeddings on Databricks.

Reads CSV files from the UC Volume, generates LLM-authored HTML documents
via the foundation model endpoint, produces real embedding vectors via
DatabricksEmbeddings, and writes results back to the volume.

Auth is handled automatically by the Databricks SDK on-cluster.

Usage (via CLI):
    uv run python -m cli upload --wheel
    uv run python -m cli upload generate_html.py
    uv run python -m cli submit generate_html.py
"""

from __future__ import annotations

import os


def main() -> None:
    from graph_feature_forge import inject_params

    inject_params()

    # Volume FUSE paths — CSVs are already uploaded via `cli upload --data`
    volume_path = os.environ.get(
        "DATABRICKS_VOLUME_PATH",
        "/Volumes/graph-feature-forge/enrichment/source-data",
    )
    csv_dir = f"{volume_path}/csv"
    html_output_dir = f"{volume_path}/html"
    embedding_output_dir = f"{volume_path}/embeddings"

    # Import the html_generator (included in the wheel)
    from html_generator.config import GeneratorConfig
    from html_generator.document_generator import generate_documents
    from html_generator.embedding_generator import (
        build_embedding_output,
        write_embedding_json,
        write_html_files,
    )

    # Build config — auth handled by databricks-sdk and databricks-langchain
    cfg = GeneratorConfig(
        llm_endpoint=os.environ.get("LLM_ENDPOINT", "databricks-claude-sonnet-4-6"),
        embedding_endpoint=os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en"),
        csv_dir=csv_dir,
        html_output_dir=html_output_dir,
        embedding_output_dir=embedding_output_dir,
        volume_base_path=volume_path,
        dry_run=False,
    )

    print("=" * 60)
    print("HTML Document & Embedding Generator (Databricks)")
    print("=" * 60)
    print(f"Volume path:     {volume_path}")
    print(f"CSV source:      {cfg.csv_dir}")
    print(f"HTML output:     {cfg.html_output_dir}")
    print(f"Embedding output:{cfg.embedding_output_dir}")
    print(f"LLM endpoint:    {cfg.llm_endpoint}")
    print(f"Embed endpoint:  {cfg.embedding_endpoint}")
    print(f"Target docs:     {cfg.total_documents}")
    print("=" * 60)

    # Step 1: Generate HTML documents via LLM
    print("\n[1/3] Generating HTML documents via LLM endpoint...")
    documents = generate_documents(cfg)

    # Step 2: Write HTML files to volume
    print("\n[2/3] Writing HTML files to volume...")
    write_html_files(documents, cfg.html_output_dir)

    # Step 3: Chunk and embed
    print("\n[3/3] Chunking and embedding via DatabricksEmbeddings...")
    embedding_output = build_embedding_output(documents, cfg)
    write_embedding_json(embedding_output, cfg.embedding_output_dir)

    print(f"\nDone: {len(documents)} HTML documents, "
          f"{embedding_output.metadata.chunk_count} embedding chunks")
    print(f"Results written to {volume_path}")


if __name__ == "__main__":
    main()
