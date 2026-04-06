"""Pydantic schemas for graph augmentation suggestions.

Pure data definitions with no DSPy dependency. These models define the
structured output that each DSPy signature produces, enabling type-safe
results that can be serialized to JSON or written directly to Neo4j.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConfidenceLevel(str, Enum):
    """Confidence level for a suggestion."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class PropertyDefinition(BaseModel):
    """Definition of a property for a node or relationship."""

    name: str = Field(..., description="Property name")
    property_type: str = Field(
        ..., description="Data type (string, int, float, boolean, date)"
    )
    required: bool = Field(default=False, description="Whether the property is required")
    description: str | None = Field(default=None, description="Description of the property")


class InvestmentTheme(BaseModel):
    """An emerging investment theme identified from analysis."""

    name: str = Field(..., description="Theme name (e.g., 'Renewable Energy', 'AI/ML')")
    description: str = Field(..., description="Description of the theme")
    market_size: str | None = Field(default=None, description="Market size or investment volume")
    growth_projection: str | None = Field(default=None, description="Growth projections")
    key_sectors: list[str] = Field(default_factory=list, description="Related sectors")
    key_companies: list[str] = Field(default_factory=list, description="Related companies")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    source_evidence: str = Field(..., description="Evidence supporting this theme")


# ---------------------------------------------------------------------------
# Suggestion types
# ---------------------------------------------------------------------------


class SuggestedNode(BaseModel):
    """A suggested new node type to add to the graph."""

    label: str = Field(..., description="Node label (e.g., 'FINANCIAL_GOAL')")
    description: str = Field(..., description="What this node type represents")
    key_property: str = Field(..., description="Property that uniquely identifies nodes")
    properties: list[PropertyDefinition] = Field(default_factory=list)
    example_values: list[dict[str, Any]] = Field(default_factory=list)
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    source_evidence: str = Field(..., description="Evidence supporting this suggestion")
    rationale: str = Field(..., description="Why this node type should be added")


class SuggestedRelationship(BaseModel):
    """A suggested new relationship type to add to the graph."""

    relationship_type: str = Field(..., description="Relationship type (e.g., 'HAS_GOAL')")
    description: str = Field(..., description="What this relationship represents")
    source_label: str = Field(..., description="Source node label")
    target_label: str = Field(..., description="Target node label")
    properties: list[PropertyDefinition] = Field(default_factory=list)
    example_instances: list[dict[str, Any]] = Field(default_factory=list)
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    source_evidence: str = Field(..., description="Evidence supporting this suggestion")
    rationale: str = Field(..., description="Why this relationship type should be added")


class SuggestedAttribute(BaseModel):
    """A suggested new attribute to add to an existing node type."""

    target_label: str = Field(..., description="Node type to add attribute to")
    property_name: str = Field(..., description="Name of the new property")
    property_type: str = Field(..., description="Data type")
    description: str = Field(..., description="What this attribute represents")
    example_values: list[Any] = Field(default_factory=list)
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    source_evidence: str = Field(..., description="Evidence supporting this suggestion")
    rationale: str = Field(..., description="Why this attribute should be added")


# ---------------------------------------------------------------------------
# Analysis results (one per signature)
# ---------------------------------------------------------------------------


class InvestmentThemesAnalysis(BaseModel):
    """Structured output for investment themes analysis."""

    summary: str = Field(..., description="Overall summary of investment themes")
    themes: list[InvestmentTheme] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class NewEntitiesAnalysis(BaseModel):
    """Structured output for new entities analysis."""

    summary: str = Field(..., description="Overall summary of suggested entities")
    suggested_nodes: list[SuggestedNode] = Field(default_factory=list)
    implementation_priority: list[str] = Field(default_factory=list)


class MissingAttributesAnalysis(BaseModel):
    """Structured output for missing attributes analysis."""

    summary: str = Field(..., description="Overall summary of missing attributes")
    suggested_attributes: list[SuggestedAttribute] = Field(default_factory=list)
    affected_node_types: list[str] = Field(default_factory=list)


class ImpliedRelationshipsAnalysis(BaseModel):
    """Structured output for implied relationships analysis."""

    summary: str = Field(..., description="Overall summary of implied relationships")
    suggested_relationships: list[SuggestedRelationship] = Field(default_factory=list)
    relationship_patterns: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Consolidated response
# ---------------------------------------------------------------------------


class AugmentationAnalysis(BaseModel):
    """Combined analysis results from all analysis types."""

    investment_themes: InvestmentThemesAnalysis | None = None
    new_entities: NewEntitiesAnalysis | None = None
    missing_attributes: MissingAttributesAnalysis | None = None
    implied_relationships: ImpliedRelationshipsAnalysis | None = None


class AugmentationResponse(BaseModel):
    """Top-level response from the augmentation agent."""

    success: bool
    analysis: AugmentationAnalysis
    all_suggested_nodes: list[SuggestedNode] = Field(default_factory=list)
    all_suggested_relationships: list[SuggestedRelationship] = Field(default_factory=list)
    all_suggested_attributes: list[SuggestedAttribute] = Field(default_factory=list)
    high_confidence_count: int = 0
    total_suggestions: int = 0

    def compute_statistics(self) -> None:
        """Recompute counts from the suggestion lists."""
        all_items = (
            self.all_suggested_nodes
            + self.all_suggested_relationships
            + self.all_suggested_attributes
        )
        self.total_suggestions = len(all_items)
        self.high_confidence_count = sum(
            1 for s in all_items if s.confidence == ConfidenceLevel.HIGH
        )
