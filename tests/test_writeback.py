"""Tests for Cypher generation and write-back."""

from semantic_auth.schemas import ConfidenceLevel, InstanceProposal, NodeReference
from semantic_auth.writeback import _cypher_literal, generate_merge_cypher


def _make_proposal(**overrides) -> InstanceProposal:
    defaults = dict(
        source_node=NodeReference(
            label="Customer", key_property="customerId", key_value="C0001",
        ),
        target_node=NodeReference(
            label="Sector", key_property="sectorId", key_value="RenewableEnergy",
        ),
        relationship_type="INTERESTED_IN",
        confidence=ConfidenceLevel.HIGH,
        source_document="customer_profile_001.txt",
        extracted_phrase="expressed interest in renewable energy",
        rationale="test",
    )
    defaults.update(overrides)
    return InstanceProposal(**defaults)


class TestCypherLiteral:
    def test_string(self):
        assert _cypher_literal("hello") == "'hello'"

    def test_string_with_quotes(self):
        assert _cypher_literal("it's") == "'it\\'s'"

    def test_string_with_backslash(self):
        assert _cypher_literal("a\\b") == "'a\\\\b'"

    def test_int(self):
        assert _cypher_literal(42) == "42"

    def test_float(self):
        assert _cypher_literal(0.92) == "0.92"

    def test_bool_true(self):
        assert _cypher_literal(True) == "true"

    def test_bool_false(self):
        assert _cypher_literal(False) == "false"

    def test_none(self):
        assert _cypher_literal(None) == "null"


class TestGenerateMergeCypher:
    def test_basic_structure(self):
        cypher = generate_merge_cypher(_make_proposal(), run_id="abc123")

        assert "MATCH (src:Customer {customerId: 'C0001'})" in cypher
        assert "MERGE (tgt:Sector {sectorId: 'RenewableEnergy'})" in cypher
        assert "MERGE (src)-[r:INTERESTED_IN]->(tgt)" in cypher

    def test_provenance_properties(self):
        cypher = generate_merge_cypher(_make_proposal(), run_id="abc123")

        assert "r.confidence = 'high'" in cypher
        assert "r.source_document = 'customer_profile_001.txt'" in cypher
        assert "r.run_id = 'abc123'" in cypher
        assert "r.enrichment_timestamp" in cypher

    def test_extracted_phrase_in_set(self):
        cypher = generate_merge_cypher(_make_proposal(), run_id="x")
        assert "r.extracted_phrase = 'expressed interest in renewable energy'" in cypher

    def test_custom_properties(self):
        p = _make_proposal(properties={"weight": 0.92})
        cypher = generate_merge_cypher(p, run_id="x")
        assert "r.weight = 0.92" in cypher

    def test_different_relationship_type(self):
        p = _make_proposal(relationship_type="CONCERNED_ABOUT")
        cypher = generate_merge_cypher(p, run_id="x")
        assert "MERGE (src)-[r:CONCERNED_ABOUT]->(tgt)" in cypher

    def test_different_labels(self):
        p = _make_proposal(
            source_node=NodeReference(label="Account", key_property="accountId", key_value="A001"),
            target_node=NodeReference(label="Goal", key_property="goalId", key_value="retirement"),
        )
        cypher = generate_merge_cypher(p, run_id="x")
        assert "MATCH (src:Account {accountId: 'A001'})" in cypher
        assert "MERGE (tgt:Goal {goalId: 'retirement'})" in cypher

    def test_special_characters_in_phrase(self):
        p = _make_proposal(extracted_phrase="he said 'renewable energy' is \"the future\"")
        cypher = generate_merge_cypher(p, run_id="x")
        # Should escape single quotes for Cypher
        assert "he said \\'renewable energy\\'" in cypher
