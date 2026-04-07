"""Neo4j write-back for approved enrichment proposals.

Generates idempotent Cypher MERGE statements from InstanceProposal
objects and executes them against Neo4j.  Each written relationship
carries provenance properties: source document, extracted phrase,
confidence level, enrichment timestamp, and run identifier.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from semantic_auth.schemas import InstanceProposal


def generate_merge_cypher(proposal: InstanceProposal, run_id: str) -> str:
    """Generate a MERGE Cypher statement for an instance proposal.

    The pattern:
    1. MATCH source node by key property
    2. MERGE target node by key property (creates if missing)
    3. MERGE the relationship with provenance properties

    MERGE is idempotent — running the same proposal twice is safe.
    """
    src = proposal.source_node
    tgt = proposal.target_node
    rel = proposal.relationship_type

    # Build provenance properties for SET clause
    props: dict[str, Any] = {
        "confidence": proposal.confidence.value,
        "source_document": proposal.source_document,
        "extracted_phrase": proposal.extracted_phrase,
        "enrichment_timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
    }
    props.update(proposal.properties)

    set_clauses = ", ".join(
        f"r.{k} = {_cypher_literal(v)}" for k, v in props.items()
    )

    return (
        f"MATCH (src:{src.label} {{{src.key_property}: {_cypher_literal(src.key_value)}}})\n"
        f"MERGE (tgt:{tgt.label} {{{tgt.key_property}: {_cypher_literal(tgt.key_value)}}})\n"
        f"MERGE (src)-[r:{rel}]->(tgt)\n"
        f"SET {set_clauses}"
    )


def _cypher_literal(value: Any) -> str:
    """Convert a Python value to a Cypher literal."""
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


class Neo4jWriter:
    """Execute enrichment proposals against Neo4j."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        import neo4j

        self._driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def write_proposals(
        self,
        proposals: list[InstanceProposal],
        run_id: str,
        dry_run: bool = True,
    ) -> list[str]:
        """Generate and optionally execute MERGE statements.

        Args:
            proposals: Instance proposals to write.
            run_id: Unique identifier for this pipeline run.
            dry_run: If True, generate Cypher but do not execute.

        Returns:
            List of generated Cypher statements.
        """
        statements = [generate_merge_cypher(p, run_id) for p in proposals]

        if dry_run:
            print(f"\n  [DRY RUN] Generated {len(statements)} Cypher statements:")
            for i, stmt in enumerate(statements, 1):
                print(f"\n  --- Statement {i} ---")
                for line in stmt.split("\n"):
                    print(f"  {line}")
            return statements

        print(f"\n  Writing {len(proposals)} proposals to Neo4j ...")
        timestamp = datetime.now(timezone.utc).isoformat()

        # Group proposals by structural pattern for UNWIND batching
        groups: dict[tuple[str, ...], list[InstanceProposal]] = {}
        for p in proposals:
            key = (
                p.source_node.label,
                p.source_node.key_property,
                p.target_node.label,
                p.target_node.key_property,
                p.relationship_type,
            )
            groups.setdefault(key, []).append(p)

        written = 0
        with self._driver.session(database=self._database) as session:
            for (src_label, src_key, tgt_label, tgt_key, rel_type), group in groups.items():
                custom_keys = sorted({k for p in group for k in p.properties})

                rows = []
                for p in group:
                    row: dict[str, Any] = {
                        "source_key": p.source_node.key_value,
                        "target_key": p.target_node.key_value,
                        "confidence": p.confidence.value,
                        "source_document": p.source_document,
                        "extracted_phrase": p.extracted_phrase,
                        "enrichment_timestamp": timestamp,
                        "run_id": run_id,
                    }
                    for k in custom_keys:
                        row[k] = p.properties.get(k)
                    rows.append(row)

                set_parts = [
                    "r.confidence = row.confidence",
                    "r.source_document = row.source_document",
                    "r.extracted_phrase = row.extracted_phrase",
                    "r.enrichment_timestamp = row.enrichment_timestamp",
                    "r.run_id = row.run_id",
                ]
                for k in custom_keys:
                    set_parts.append(f"r.{k} = row.{k}")

                query = (
                    "UNWIND $rows AS row "
                    f"MATCH (src:{src_label} {{{src_key}: row.source_key}}) "
                    f"MERGE (tgt:{tgt_label} {{{tgt_key}: row.target_key}}) "
                    f"MERGE (src)-[r:{rel_type}]->(tgt) "
                    f"SET {', '.join(set_parts)}"
                )

                try:
                    session.run(query, {"rows": rows})
                    written += len(group)
                except Exception as exc:
                    print(f"    {rel_type} ({len(group)} proposals) FAILED: {exc}")

        print(f"  Write complete: {written}/{len(proposals)} succeeded")
        return statements

    def __enter__(self) -> Neo4jWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
