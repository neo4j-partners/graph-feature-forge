"""Delta table for tracking enrichment proposals across pipeline runs.

Stores every enrichment proposal written to Neo4j so that subsequent
runs can deduplicate, provide prior-enrichment context to the LLM,
and maintain an audit trail.

The enrichment log is a single Delta table with ``relationship_type``
as a column and a JSON column for custom properties — no DDL needed
when the LLM proposes new relationship types.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from semantic_auth.schemas import InstanceProposal
from semantic_auth.structured_data import SQLExecutor


class EnrichmentStore:
    """Read/write interface to the ``enrichment_log`` Delta table."""

    TABLE_NAME = "enrichment_log"

    def __init__(
        self,
        execute_sql: SQLExecutor,
        catalog: str,
        schema: str,
    ) -> None:
        self._execute = execute_sql
        self._catalog = catalog
        self._schema = schema
        self._fq_table = f"`{catalog}`.`{schema}`.`{self.TABLE_NAME}`"

    # ------------------------------------------------------------------
    # DDL
    # ------------------------------------------------------------------

    def ensure_table(self) -> None:
        """Create the enrichment_log table if it doesn't exist."""
        self._execute(
            f"""\
CREATE TABLE IF NOT EXISTS {self._fq_table} (
    run_id STRING,
    enrichment_timestamp TIMESTAMP,
    relationship_type STRING,
    source_label STRING,
    source_key_property STRING,
    source_key_value STRING,
    target_label STRING,
    target_key_property STRING,
    target_key_value STRING,
    confidence STRING,
    source_document STRING,
    extracted_phrase STRING,
    rationale STRING,
    properties_json STRING
)"""
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_proposals(
        self,
        proposals: list[InstanceProposal],
        run_id: str,
    ) -> int:
        """Insert proposals into the enrichment log.

        Returns the number of rows written.
        """
        if not proposals:
            return 0

        timestamp = datetime.now(timezone.utc).isoformat()

        values_rows: list[str] = []
        for p in proposals:
            props_json = json.dumps(p.properties) if p.properties else "{}"
            values_rows.append(
                "("
                + ", ".join(
                    [
                        _sql_str(run_id),
                        f"TIMESTAMP '{timestamp}'",
                        _sql_str(p.relationship_type),
                        _sql_str(p.source_node.label),
                        _sql_str(p.source_node.key_property),
                        _sql_str(p.source_node.key_value),
                        _sql_str(p.target_node.label),
                        _sql_str(p.target_node.key_property),
                        _sql_str(p.target_node.key_value),
                        _sql_str(p.confidence.value),
                        _sql_str(p.source_document),
                        _sql_str(p.extracted_phrase),
                        _sql_str(p.rationale),
                        _sql_str(props_json),
                    ]
                )
                + ")"
            )

        sql = (
            f"INSERT INTO {self._fq_table} VALUES\n"
            + ",\n".join(values_rows)
        )
        self._execute(sql)
        return len(proposals)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_existing_keys(self) -> set[tuple[str, str, str, str, str]]:
        """Return the set of (rel_type, src_label, src_key_value, tgt_label, tgt_key_value).

        Used for bulk deduplication before writing new proposals.
        """
        rows = self._execute(
            f"""\
SELECT DISTINCT
    relationship_type,
    source_label,
    source_key_value,
    target_label,
    target_key_value
FROM {self._fq_table}"""
        )
        return {
            (
                r["relationship_type"],
                r["source_label"],
                r["source_key_value"],
                r["target_label"],
                r["target_key_value"],
            )
            for r in rows
        }

    def format_context(self) -> str:
        """Format prior enrichments as an LLM context string.

        Returns a readable summary grouped by relationship type,
        or an empty string if no enrichments exist.
        """
        rows = self._execute(
            f"""\
SELECT
    relationship_type,
    source_label,
    source_key_value,
    target_label,
    target_key_value,
    confidence
FROM {self._fq_table}
ORDER BY relationship_type, source_key_value"""
        )

        if not rows:
            return ""

        # Group by relationship type
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(r["relationship_type"], []).append(r)

        lines = ["## Prior Enrichments (already written to graph)\n"]
        for rel_type, entries in sorted(groups.items()):
            lines.append(f"### {rel_type} ({len(entries)} relationships)")
            for e in entries:
                lines.append(
                    f"- ({e['source_label']} {e['source_key_value']}) "
                    f"-[{rel_type}]-> "
                    f"({e['target_label']} {e['target_key_value']}) "
                    f"[{e['confidence']}]"
                )
            lines.append("")

        lines.append(
            "Do NOT re-propose relationships that already appear above. "
            "Focus on discovering new gaps and patterns.\n"
        )
        return "\n".join(lines)

    def deduplicate(
        self, proposals: list[InstanceProposal],
    ) -> list[InstanceProposal]:
        """Remove proposals that already exist in the enrichment log.

        Returns only the proposals whose dedup_key is not already stored.
        """
        existing = self.get_existing_keys()
        if not existing:
            return proposals
        return [p for p in proposals if p.dedup_key not in existing]

    def count(self) -> int:
        """Return total number of enrichment rows."""
        rows = self._execute(
            f"SELECT COUNT(*) AS cnt FROM {self._fq_table}"
        )
        return rows[0]["cnt"] if rows else 0


def _sql_str(value: str) -> str:
    """Escape a string for inclusion as a Databricks SQL string literal.

    Handles backslashes, single quotes, and null bytes.  Inputs are
    Pydantic-validated ``InstanceProposal`` fields — this function is
    a defence-in-depth measure, not a general-purpose injection guard.
    """
    escaped = value.replace("\\", "\\\\").replace("'", "''").replace("\x00", "")
    return f"'{escaped}'"
