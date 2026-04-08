"""Tests for reporting output functions."""

from graph_feature_forge.schemas import (
    ConfidenceLevel,
    FilteredProposals,
    InstanceProposal,
    NodeReference,
)
from graph_feature_forge.reporting import print_filtered_proposals, print_instance_proposal


def _make_proposal(confidence: ConfidenceLevel = ConfidenceLevel.HIGH) -> InstanceProposal:
    return InstanceProposal(
        source_node=NodeReference(label="Customer", key_property="customerId", key_value="C0001"),
        target_node=NodeReference(label="Sector", key_property="sectorId", key_value="RenewableEnergy"),
        relationship_type="INTERESTED_IN",
        confidence=confidence,
        source_document="customer_profile_001.txt",
        extracted_phrase="expressed interest in renewable energy",
        rationale="test rationale",
    )


class TestPrintInstanceProposal:
    def test_does_not_crash(self, capsys):
        print_instance_proposal(_make_proposal(), 1)
        captured = capsys.readouterr()
        assert "Customer C0001" in captured.out
        assert "INTERESTED_IN" in captured.out
        assert "Sector RenewableEnergy" in captured.out


class TestPrintFilteredProposals:
    def test_empty(self, capsys):
        f = FilteredProposals.from_proposals([])
        print_filtered_proposals(f)
        captured = capsys.readouterr()
        assert "Total proposals: 0" in captured.out

    def test_mixed(self, capsys):
        proposals = [
            _make_proposal(ConfidenceLevel.HIGH),
            _make_proposal(ConfidenceLevel.MEDIUM),
            _make_proposal(ConfidenceLevel.LOW),
        ]
        f = FilteredProposals.from_proposals(proposals)
        print_filtered_proposals(f)
        captured = capsys.readouterr()
        assert "AUTO-APPROVE (HIGH):  1" in captured.out
        assert "FLAGGED (MEDIUM):     1" in captured.out
        assert "REVIEW (LOW):         1" in captured.out
        assert "AUTO-APPROVE" in captured.out
        assert "will write to Neo4j" in captured.out
