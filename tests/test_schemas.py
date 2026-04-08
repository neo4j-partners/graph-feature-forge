"""Tests for schemas: InstanceProposal, FilteredProposals, NodeReference."""

from graph_feature_forge.analysis.schemas import (
    ConfidenceLevel,
    FilteredProposals,
    InstanceProposal,
    InstanceResolutionResult,
    NodeReference,
)


def _make_proposal(
    source_id: str = "C0001",
    target_id: str = "RenewableEnergy",
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
) -> InstanceProposal:
    return InstanceProposal(
        source_node=NodeReference(
            label="Customer", key_property="customerId", key_value=source_id,
        ),
        target_node=NodeReference(
            label="Sector", key_property="sectorId", key_value=target_id,
        ),
        relationship_type="INTERESTED_IN",
        confidence=confidence,
        source_document="customer_profile_001.txt",
        extracted_phrase="expressed interest in renewable energy",
        rationale="Customer profile explicitly states interest",
    )


class TestNodeReference:
    def test_creation(self):
        ref = NodeReference(label="Customer", key_property="customerId", key_value="C0001")
        assert ref.label == "Customer"
        assert ref.key_property == "customerId"
        assert ref.key_value == "C0001"

    def test_serialization(self):
        ref = NodeReference(label="Customer", key_property="customerId", key_value="C0001")
        d = ref.model_dump()
        assert d == {"label": "Customer", "key_property": "customerId", "key_value": "C0001"}


class TestInstanceProposal:
    def test_creation(self):
        p = _make_proposal()
        assert p.source_node.key_value == "C0001"
        assert p.target_node.key_value == "RenewableEnergy"
        assert p.relationship_type == "INTERESTED_IN"
        assert p.confidence == ConfidenceLevel.HIGH

    def test_with_properties(self):
        p = _make_proposal()
        p.properties["weight"] = 0.92
        assert p.properties["weight"] == 0.92

    def test_json_roundtrip(self):
        p = _make_proposal()
        json_str = p.model_dump_json()
        p2 = InstanceProposal.model_validate_json(json_str)
        assert p2.source_node.key_value == p.source_node.key_value
        assert p2.confidence == p.confidence

    def test_dedup_key(self):
        p = _make_proposal()
        assert p.dedup_key == (
            "INTERESTED_IN", "Customer", "C0001", "Sector", "RenewableEnergy",
        )

    def test_dedup_key_differs_by_target(self):
        p1 = _make_proposal()
        p2 = _make_proposal()
        p2.target_node.key_value = "Technology"
        assert p1.dedup_key != p2.dedup_key


class TestInstanceResolutionResult:
    def test_empty(self):
        r = InstanceResolutionResult(proposals=[], resolution_summary="nothing found")
        assert len(r.proposals) == 0

    def test_with_proposals(self):
        r = InstanceResolutionResult(
            proposals=[_make_proposal(), _make_proposal(source_id="C0002")],
            resolution_summary="found 2 proposals",
        )
        assert len(r.proposals) == 2


class TestFilteredProposals:
    def test_empty(self):
        f = FilteredProposals.from_proposals([])
        assert len(f.auto_approve) == 0
        assert len(f.flagged) == 0
        assert len(f.review) == 0

    def test_all_high(self):
        proposals = [_make_proposal(confidence=ConfidenceLevel.HIGH) for _ in range(3)]
        f = FilteredProposals.from_proposals(proposals)
        assert len(f.auto_approve) == 3
        assert len(f.flagged) == 0
        assert len(f.review) == 0

    def test_all_low(self):
        proposals = [_make_proposal(confidence=ConfidenceLevel.LOW) for _ in range(2)]
        f = FilteredProposals.from_proposals(proposals)
        assert len(f.auto_approve) == 0
        assert len(f.flagged) == 0
        assert len(f.review) == 2

    def test_mixed_confidence(self):
        proposals = [
            _make_proposal(source_id="C0001", confidence=ConfidenceLevel.HIGH),
            _make_proposal(source_id="C0002", confidence=ConfidenceLevel.MEDIUM),
            _make_proposal(source_id="C0003", confidence=ConfidenceLevel.LOW),
            _make_proposal(source_id="C0004", confidence=ConfidenceLevel.HIGH),
        ]
        f = FilteredProposals.from_proposals(proposals)
        assert len(f.auto_approve) == 2
        assert len(f.flagged) == 1
        assert len(f.review) == 1

    def test_preserves_order(self):
        proposals = [
            _make_proposal(source_id="C0001", confidence=ConfidenceLevel.HIGH),
            _make_proposal(source_id="C0002", confidence=ConfidenceLevel.HIGH),
        ]
        f = FilteredProposals.from_proposals(proposals)
        assert f.auto_approve[0].source_node.key_value == "C0001"
        assert f.auto_approve[1].source_node.key_value == "C0002"
