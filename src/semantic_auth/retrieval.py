"""Document retrieval via in-memory cosine similarity.

Loads pre-computed embeddings from the workshop's document chunks and
performs similarity search at query time using databricks-gte-large-en.

The corpus is small (20 chunks, 14 documents) so in-memory search is
appropriate. The interface is designed for backend swapability — replace
the retrieval internals with Neo4j vector index or Mosaic AI Vector
Search without changing callers.

Embedding file format (``document_chunks_embedded.json``)::

    {
        "metadata": { "embedding_model": "...", "chunk_count": 20, ... },
        "documents": [ { "document_id": "...", "filename": "...", ... } ],
        "chunks": [
            {
                "chunk_id": "uuid",
                "index": 0,
                "text": "...",
                "document_id": "uuid",
                "metadata": { "document_title": "...", "document_type": "...", ... },
                "embedding": [0.01, -0.03, ...]
            },
            ...
        ]
    }
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable

Embedder = Callable[[str], list[float]]


@dataclass
class RetrievedChunk:
    """A document chunk returned by similarity search."""

    chunk_id: str
    text: str
    score: float
    document_id: str
    metadata: dict[str, Any]


class DocumentRetrieval:
    """Retrieve relevant document chunks via embedding similarity.

    Args:
        chunks: List of chunk dicts, each with ``text``, ``embedding``,
            ``chunk_id``, ``document_id``, and ``metadata`` keys.
        embedder: Callable that takes a text string and returns its
            embedding vector as a list of floats. Use
            ``make_sdk_embedder`` to create one.
    """

    def __init__(self, chunks: list[dict], embedder: Embedder) -> None:
        self._chunks = chunks
        self._embedder = embedder

    @classmethod
    def from_json_path(cls, path: str, embedder: Embedder) -> DocumentRetrieval:
        """Load chunks from a JSON file on disk or a UC volume path.

        On Databricks clusters, UC volume paths are accessible as
        regular file paths (e.g.,
        ``/Volumes/catalog/schema/volume/file.json``).
        """
        with open(path) as f:
            data = json.load(f)
        return cls(data["chunks"], embedder)

    @classmethod
    def from_json_str(cls, raw: str, embedder: Embedder) -> DocumentRetrieval:
        """Load chunks from a raw JSON string."""
        data = json.loads(raw)
        return cls(data["chunks"], embedder)

    def query(self, text: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Find the top-k most similar chunks to the query text."""
        query_embedding = self._embedder(text)

        scored = []
        for chunk in self._chunks:
            score = _cosine_similarity(query_embedding, chunk["embedding"])
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                score=score,
                document_id=chunk["document_id"],
                metadata=chunk.get("metadata", {}),
            )
            for score, chunk in scored[:top_k]
        ]

    def format_context(self, text: str, top_k: int = 5) -> str:
        """Query and format results as an LLM context string.

        Returns a markdown-formatted string with numbered document
        excerpts, titles, types, and relevance scores — ready to be
        included in a synthesis prompt.
        """
        chunks = self.query(text, top_k)
        lines = ["# Retrieved Document Excerpts", ""]

        if not chunks:
            lines.append("No relevant documents found.")
            return "\n".join(lines)

        for i, chunk in enumerate(chunks, 1):
            title = chunk.metadata.get("document_title", "Unknown")
            doc_type = chunk.metadata.get("document_type", "unknown")
            lines.append(
                f"## [{i}] {title} ({doc_type}, relevance: {chunk.score:.3f})"
            )
            lines.append("")
            lines.append(chunk.text)
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Embedder factories
# ---------------------------------------------------------------------------


def make_sdk_embedder(endpoint: str = "databricks-gte-large-en") -> Embedder:
    """Create an embedder using the Databricks SDK.

    Works both locally (with ``DATABRICKS_HOST``/``DATABRICKS_TOKEN``)
    and on-cluster (with automatic credentials).
    """
    from databricks.sdk import WorkspaceClient

    wc = WorkspaceClient()

    def embed(text: str) -> list[float]:
        response = wc.serving_endpoints.query(
            name=endpoint,
            input=[text],
        )
        return response.data[0].embedding

    return embed


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
