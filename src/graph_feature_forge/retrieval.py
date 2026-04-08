"""Document retrieval via embedding similarity.

Two backends are available:

- **DocumentRetrieval** — in-memory cosine similarity over pre-computed
  embeddings loaded from a JSON file.  Suitable for small corpora and
  environments without Neo4j access.

- **Neo4jRetrieval** — queries Neo4j's ``chunk_embedding_index`` vector
  index and optionally traverses graph relationships
  (``FROM_DOCUMENT → DESCRIBES → Customer``) to return entity context
  alongside matched chunks.

Both classes expose the same ``query()`` / ``format_context()`` interface
so callers (e.g., ``GapAnalysisSynthesizer``) work with either backend.

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


class Neo4jRetrieval:
    """Retrieve relevant document chunks via Neo4j vector index.

    Queries the ``chunk_embedding_index`` vector index and traverses
    graph relationships to return entity context alongside matched
    chunks.  The traversal follows::

        (Chunk)-[:FROM_DOCUMENT]->(Document)-[:DESCRIBES]->(Customer)

    This gives the synthesis step a direct path from unstructured
    evidence to the entity it concerns, eliminating misattribution
    errors.

    Args:
        driver: An open ``neo4j.Driver`` instance.
        embedder: Callable that takes a text string and returns its
            embedding vector as a list of floats.
        database: Neo4j database name (default ``"neo4j"``).
    """

    def __init__(
        self,
        driver: Any,
        embedder: Embedder,
        database: str = "neo4j",
    ) -> None:
        self._driver = driver
        self._embedder = embedder
        self._database = database

    _GRAPH_AWARE_QUERY = """\
CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $embedding)
YIELD node AS chunk, score
OPTIONAL MATCH (chunk)-[:FROM_DOCUMENT]->(doc:Document)
OPTIONAL MATCH (doc)-[:DESCRIBES]->(c:Customer)
RETURN chunk.chunk_id AS chunk_id,
       chunk.text AS text,
       score,
       doc.document_id AS document_id,
       doc.title AS document_title,
       doc.document_type AS document_type,
       c.customer_id AS customer_id,
       c.first_name + ' ' + c.last_name AS customer_name
ORDER BY score DESC
"""

    def query(self, text: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Find the top-k most similar chunks using Neo4j vector search."""
        query_embedding = self._embedder(text)

        def _run(tx: Any) -> list[Any]:
            result = tx.run(
                self._GRAPH_AWARE_QUERY,
                embedding=query_embedding,
                top_k=top_k,
            )
            return list(result)

        with self._driver.session(database=self._database) as session:
            records = session.execute_read(_run)

        return [
            RetrievedChunk(
                chunk_id=record["chunk_id"],
                text=record["text"],
                score=record["score"],
                document_id=record["document_id"] or "",
                metadata={
                    "document_title": record["document_title"] or "Unknown",
                    "document_type": record["document_type"] or "unknown",
                    "customer_id": record["customer_id"],
                    "customer_name": record["customer_name"],
                },
            )
            for record in records
        ]

    def format_context(self, text: str, top_k: int = 5) -> str:
        """Query and format results as an LLM context string.

        Returns a markdown-formatted string with numbered document
        excerpts, titles, types, relevance scores, and customer
        attribution when available.
        """
        chunks = self.query(text, top_k)
        lines = ["# Retrieved Document Excerpts", ""]

        if not chunks:
            lines.append("No relevant documents found.")
            return "\n".join(lines)

        for i, chunk in enumerate(chunks, 1):
            title = chunk.metadata.get("document_title", "Unknown")
            doc_type = chunk.metadata.get("document_type", "unknown")
            header = f"## [{i}] {title} ({doc_type}, relevance: {chunk.score:.3f})"

            customer_name = chunk.metadata.get("customer_name")
            customer_id = chunk.metadata.get("customer_id")
            if customer_name and customer_id:
                header += f" — Customer: {customer_name} ({customer_id})"

            lines.append(header)
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
