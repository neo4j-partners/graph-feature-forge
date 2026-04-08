"""Chunk HTML documents and generate embedding vectors.

In live mode, uses ``DatabricksEmbeddings`` from ``databricks-langchain``
to produce real 1024-dim vectors (auth handled automatically on-cluster).
In dry-run mode, generates random vectors for local testing.

Chunking uses a two-pass langchain approach:
1. ``HTMLHeaderTextSplitter`` splits by h1/h2/h3 headers, preserving
   section metadata on each chunk.
2. ``RecursiveCharacterTextSplitter`` enforces max chunk size with overlap.
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import List, Tuple

from .config import GeneratorConfig
from .models import (
    ChunkMetadata,
    ChunkRecord,
    DocumentRecord,
    EmbeddingOutput,
    GeneratedDocument,
)


# ---------------------------------------------------------------------------
# HTML → chunks (langchain two-pass: header split then size split)
# ---------------------------------------------------------------------------

# Headers to split on — metadata keys match ChunkMetadata fields
HEADERS_TO_SPLIT_ON = [
    ("h1", "section_header"),
    ("h2", "section_subheader"),
    ("h3", "section_detail"),
]

MIN_CHUNK_LENGTH = 50  # Filter out tiny header-only fragments


def chunk_html(
    html: str,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
) -> List[Tuple[str, dict]]:
    """Split HTML into overlapping text chunks with header metadata.

    Returns a list of ``(text, metadata_dict)`` tuples where metadata
    contains ``section_header``, ``section_subheader``, and
    ``section_detail`` keys extracted from the HTML header hierarchy.
    """
    from langchain_text_splitters import (
        HTMLHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    # Pass 1: split by HTML headers, preserving section metadata
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    header_docs = html_splitter.split_text(html)

    # Pass 2: enforce max chunk size with overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    final_docs = text_splitter.split_documents(header_docs)

    # Extract (text, metadata) pairs, filtering tiny fragments
    results: List[Tuple[str, dict]] = []
    for doc in final_docs:
        text = doc.page_content.strip()
        if len(text) >= MIN_CHUNK_LENGTH:
            results.append((text, doc.metadata))

    return results


# ---------------------------------------------------------------------------
# Embedding — live vs dry-run
# ---------------------------------------------------------------------------


def _embed_batch_live(
    texts: List[str],
    cfg: GeneratorConfig,
) -> List[List[float]]:
    """Embed texts using DatabricksEmbeddings (auth handled automatically).

    Uses ``databricks-langchain`` which picks up workspace credentials
    automatically on-cluster, matching the full-demo reference pattern.
    """
    from databricks_langchain import DatabricksEmbeddings

    embedder = DatabricksEmbeddings(endpoint=cfg.embedding_endpoint)
    return embedder.embed_documents(texts)


def _embed_batch_dry_run(
    texts: List[str],
    cfg: GeneratorConfig,
    rng: random.Random,
) -> List[List[float]]:
    """Generate random vectors that match the expected dimensionality."""
    dim = cfg.embedding_dimensions
    return [[rng.gauss(0, 0.5) for _ in range(dim)] for _ in texts]


def embed_texts(
    texts: List[str],
    cfg: GeneratorConfig,
    rng: random.Random,
    batch_size: int = 16,
) -> List[List[float]]:
    """Embed a list of texts, batching as needed.

    In dry-run mode, returns random vectors.  In live mode, calls the
    Databricks endpoint via ``DatabricksEmbeddings`` in batches of
    *batch_size* to avoid rate limits.
    """
    if cfg.dry_run:
        return _embed_batch_dry_run(texts, cfg, rng)

    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1} "
              f"({len(batch)} texts)...")
        embeddings = _embed_batch_live(batch, cfg)
        all_embeddings.extend(embeddings)
        if i + batch_size < len(texts):
            print(f"  Embedded {i + batch_size}/{len(texts)} chunks...")
    return all_embeddings


# ---------------------------------------------------------------------------
# Build the full output structure
# ---------------------------------------------------------------------------


def build_embedding_output(
    documents: List[GeneratedDocument],
    cfg: GeneratorConfig,
) -> EmbeddingOutput:
    """Chunk documents, embed them, and assemble the output JSON object.

    Returns an ``EmbeddingOutput`` that can be serialised directly to
    ``document_chunks_embedded.json``.
    """
    rng = random.Random(cfg.seed + 1)  # Different seed than doc generation

    doc_records: List[DocumentRecord] = []
    chunk_records: List[ChunkRecord] = []
    all_chunk_texts: List[str] = []

    for doc_idx, doc in enumerate(documents):
        doc_id = str(uuid.uuid4())
        chunks = chunk_html(doc.html_content, cfg.chunk_size, cfg.chunk_overlap)

        source_path = f"{cfg.volume_base_path}/html/{doc.filename}"

        # Compute total char count from all chunk texts
        total_chars = sum(len(text) for text, _ in chunks)

        doc_records.append(
            DocumentRecord(
                document_id=doc_id,
                filename=doc.filename,
                document_type=doc.document_type,
                title=doc.title,
                source_path=source_path,
                char_count=total_chars,
                entity_references=doc.entity_references,
            )
        )

        for chunk_idx, (chunk_text_str, header_meta) in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            all_chunk_texts.append(chunk_text_str)
            chunk_records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    index=chunk_idx,
                    text=chunk_text_str,
                    document_id=doc_id,
                    metadata=ChunkMetadata(
                        document_title=doc.title,
                        document_type=doc.document_type,
                        source_path=source_path,
                        section_header=header_meta.get("section_header", ""),
                        section_subheader=header_meta.get("section_subheader", ""),
                        section_detail=header_meta.get("section_detail", ""),
                    ),
                    embedding=[],  # Placeholder — filled below
                )
            )

    # Generate embeddings for all chunks
    print(f"Embedding {len(all_chunk_texts)} chunks...")
    embeddings = embed_texts(all_chunk_texts, cfg, rng)

    # Fill in the embedding vectors
    for i, emb in enumerate(embeddings):
        chunk_records[i].embedding = emb

    meta = EmbeddingOutput.build_metadata(
        model_name=cfg.embedding_endpoint,
        dimensions=cfg.embedding_dimensions,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        document_count=len(doc_records),
        chunk_count=len(chunk_records),
    )

    print(f"Built {len(doc_records)} document records, "
          f"{len(chunk_records)} chunk records")

    return EmbeddingOutput(
        metadata=meta,
        documents=doc_records,
        chunks=chunk_records,
    )


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------


def write_html_files(
    documents: List[GeneratedDocument],
    output_dir: Path,
) -> None:
    """Write each document's HTML content to a file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for doc in documents:
        path = output_dir / doc.filename
        path.write_text(doc.html_content, encoding="utf-8")
    print(f"Wrote {len(documents)} HTML files to {output_dir}")


def write_embedding_json(
    output: EmbeddingOutput,
    output_dir: Path,
) -> None:
    """Write the embedding output JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "document_chunks_embedded.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(output.model_dump(), fh, indent=2)
    print(f"Wrote embedding JSON to {path}")
