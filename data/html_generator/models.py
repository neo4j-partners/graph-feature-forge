"""Pydantic models for documents, chunks, and the embedding output schema.

These models enforce the exact JSON structure produced by the existing
embedding pipeline in ``data/embeddings/document_chunks_embedded.json``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Entity references embedded inside each document record
# ---------------------------------------------------------------------------

class EntityReferences(BaseModel):
    """Cross-references to entities from the CSV data."""

    customers: List[str] = Field(default_factory=list)
    companies: List[str] = Field(default_factory=list)
    stock_tickers: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Document-level metadata (appears in the top-level ``documents`` array)
# ---------------------------------------------------------------------------

class DocumentRecord(BaseModel):
    """One entry in the ``documents`` array of the output JSON."""

    document_id: str
    filename: str
    document_type: str
    title: str
    source_path: str
    char_count: int
    entity_references: EntityReferences


# ---------------------------------------------------------------------------
# Per-chunk metadata (nested inside each chunk)
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """Metadata block inside a single chunk."""

    document_title: str
    document_type: str
    source_path: str
    section_header: str = ""
    section_subheader: str = ""
    section_detail: str = ""


class ChunkRecord(BaseModel):
    """One entry in the ``chunks`` array of the output JSON."""

    chunk_id: str
    index: int
    text: str
    document_id: str
    metadata: ChunkMetadata
    embedding: List[float]


# ---------------------------------------------------------------------------
# Top-level output schema
# ---------------------------------------------------------------------------

class EmbeddingMetadata(BaseModel):
    """Top-level ``metadata`` block in the output JSON."""

    generated_at: str
    embedding_model: str
    embedding_dimensions: int
    chunk_size: int
    chunk_overlap: int
    document_count: int
    chunk_count: int


class EmbeddingOutput(BaseModel):
    """Root schema for ``document_chunks_embedded.json``."""

    metadata: EmbeddingMetadata
    documents: List[DocumentRecord]
    chunks: List[ChunkRecord]

    @classmethod
    def build_metadata(
        cls,
        *,
        model_name: str,
        dimensions: int,
        chunk_size: int,
        chunk_overlap: int,
        document_count: int,
        chunk_count: int,
    ) -> EmbeddingMetadata:
        return EmbeddingMetadata(
            generated_at=datetime.now(timezone.utc).isoformat(),
            embedding_model=model_name,
            embedding_dimensions=dimensions,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            document_count=document_count,
            chunk_count=chunk_count,
        )


# ---------------------------------------------------------------------------
# Internal models used during generation (not serialised to JSON)
# ---------------------------------------------------------------------------

class GeneratedDocument(BaseModel):
    """Intermediate representation of an HTML document before embedding."""

    filename: str
    document_type: str
    title: str
    html_content: str
    entity_references: EntityReferences
