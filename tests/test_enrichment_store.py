"""Tests for the enrichment store."""

from graph_feature_forge.data.enrichment_store import EnrichmentStore, _sql_str
from graph_feature_forge.analysis.schemas import ConfidenceLevel, InstanceProposal, NodeReference


def _make_proposal(
    rel_type: str = "INTERESTED_IN",
    src_label: str = "Customer",
    src_key: str = "customer_id",
    src_value: str = "C0001",
    tgt_label: str = "Sector",
    tgt_key: str = "sector_id",
    tgt_value: str = "RenewableEnergy",
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
) -> InstanceProposal:
    return InstanceProposal(
        source_node=NodeReference(label=src_label, key_property=src_key, key_value=src_value),
        target_node=NodeReference(label=tgt_label, key_property=tgt_key, key_value=tgt_value),
        relationship_type=rel_type,
        properties={},
        confidence=confidence,
        source_document="test_doc.html",
        extracted_phrase="interested in renewable energy",
        rationale="Test rationale",
    )


class TestSqlStr:
    def test_simple_string(self):
        assert _sql_str("hello") == "'hello'"

    def test_escapes_single_quotes(self):
        assert _sql_str("it's") == "'it''s'"

    def test_empty_string(self):
        assert _sql_str("") == "''"

    def test_escapes_backslashes(self):
        assert _sql_str("a\\b") == "'a\\\\b'"

    def test_strips_null_bytes(self):
        assert _sql_str("a\x00b") == "'ab'"

    def test_handles_combined_special_chars(self):
        assert _sql_str("it's a\\b") == "'it''s a\\\\b'"


class TestEnsureTable:
    def test_generates_create_table_sql(self):
        captured: list[str] = []

        def fake_executor(sql: str):
            captured.append(sql)
            return []

        store = EnrichmentStore(fake_executor, "cat", "sch")
        store.ensure_table()

        assert len(captured) == 1
        sql = captured[0]
        assert "CREATE TABLE IF NOT EXISTS" in sql
        assert "`cat`.`sch`.`enrichment_log`" in sql
        assert "run_id STRING" in sql
        assert "enrichment_timestamp TIMESTAMP" in sql
        assert "relationship_type STRING" in sql
        assert "properties_json STRING" in sql


class TestWriteProposals:
    def test_generates_insert_sql(self):
        captured: list[str] = []

        def fake_executor(sql: str):
            captured.append(sql)
            return []

        store = EnrichmentStore(fake_executor, "cat", "sch")
        proposal = _make_proposal()
        count = store.write_proposals([proposal], run_id="abc123")

        assert count == 1
        assert len(captured) == 1
        sql = captured[0]
        assert "INSERT INTO `cat`.`sch`.`enrichment_log`" in sql
        assert "'abc123'" in sql
        assert "'INTERESTED_IN'" in sql
        assert "'Customer'" in sql
        assert "'C0001'" in sql
        assert "'Sector'" in sql
        assert "'RenewableEnergy'" in sql
        assert "'high'" in sql

    def test_empty_proposals_returns_zero(self):
        captured: list[str] = []

        def fake_executor(sql: str):
            captured.append(sql)
            return []

        store = EnrichmentStore(fake_executor, "cat", "sch")
        count = store.write_proposals([], run_id="abc123")

        assert count == 0
        assert len(captured) == 0


class TestGetExistingKeys:
    def test_parses_rows_into_tuples(self):
        fake_rows = [
            {
                "relationship_type": "INTERESTED_IN",
                "source_label": "Customer",
                "source_key_value": "C0001",
                "target_label": "Sector",
                "target_key_value": "RenewableEnergy",
            },
            {
                "relationship_type": "CONCERNED_ABOUT",
                "source_label": "Customer",
                "source_key_value": "C0002",
                "target_label": "Risk",
                "target_key_value": "MarketVolatility",
            },
        ]

        def fake_executor(sql: str):
            return fake_rows

        store = EnrichmentStore(fake_executor, "cat", "sch")
        keys = store.get_existing_keys()

        assert len(keys) == 2
        assert ("INTERESTED_IN", "Customer", "C0001", "Sector", "RenewableEnergy") in keys
        assert ("CONCERNED_ABOUT", "Customer", "C0002", "Risk", "MarketVolatility") in keys

    def test_empty_table_returns_empty_set(self):
        def fake_executor(sql: str):
            return []

        store = EnrichmentStore(fake_executor, "cat", "sch")
        keys = store.get_existing_keys()
        assert keys == set()


class TestFormatContext:
    def test_empty_returns_empty_string(self):
        def fake_executor(sql: str):
            return []

        store = EnrichmentStore(fake_executor, "cat", "sch")
        assert store.format_context() == ""

    def test_formats_grouped_by_relationship_type(self):
        fake_rows = [
            {
                "relationship_type": "INTERESTED_IN",
                "source_label": "Customer",
                "source_key_value": "C0001",
                "target_label": "Sector",
                "target_key_value": "RenewableEnergy",
                "confidence": "high",
            },
            {
                "relationship_type": "INTERESTED_IN",
                "source_label": "Customer",
                "source_key_value": "C0002",
                "target_label": "Sector",
                "target_key_value": "Technology",
                "confidence": "medium",
            },
        ]

        def fake_executor(sql: str):
            return fake_rows

        store = EnrichmentStore(fake_executor, "cat", "sch")
        context = store.format_context()

        assert "## Prior Enrichments" in context
        assert "### INTERESTED_IN (2 relationships)" in context
        assert "(Customer C0001) -[INTERESTED_IN]-> (Sector RenewableEnergy) [high]" in context
        assert "(Customer C0002) -[INTERESTED_IN]-> (Sector Technology) [medium]" in context
        assert "Do NOT re-propose" in context


class TestDeduplicate:
    def test_removes_existing_proposals(self):
        fake_rows = [
            {
                "relationship_type": "INTERESTED_IN",
                "source_label": "Customer",
                "source_key_value": "C0001",
                "target_label": "Sector",
                "target_key_value": "RenewableEnergy",
            },
        ]

        def fake_executor(sql: str):
            return fake_rows

        store = EnrichmentStore(fake_executor, "cat", "sch")

        existing = _make_proposal()  # matches the fake row
        new = _make_proposal(tgt_value="Technology")  # does not match
        result = store.deduplicate([existing, new])

        assert len(result) == 1
        assert result[0].target_node.key_value == "Technology"

    def test_keeps_all_when_none_exist(self):
        def fake_executor(sql: str):
            return []

        store = EnrichmentStore(fake_executor, "cat", "sch")
        proposals = [_make_proposal(), _make_proposal(tgt_value="Technology")]
        result = store.deduplicate(proposals)
        assert len(result) == 2


class TestCount:
    def test_returns_count(self):
        def fake_executor(sql: str):
            return [{"cnt": 42}]

        store = EnrichmentStore(fake_executor, "cat", "sch")
        assert store.count() == 42

    def test_returns_zero_on_empty(self):
        def fake_executor(sql: str):
            return []

        store = EnrichmentStore(fake_executor, "cat", "sch")
        assert store.count() == 0
