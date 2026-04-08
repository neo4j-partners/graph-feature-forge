"""Result formatting and PASS/FAIL validation harness.

Separated from business logic so the analyzers stay focused on DSPy
and this module stays focused on presentation.
"""

from __future__ import annotations

from graph_feature_forge.analyzers import AnalysisResult
from graph_feature_forge.schemas import (
    AugmentationResponse,
    ConfidenceLevel,
    FilteredProposals,
    ImpliedRelationshipsAnalysis,
    InstanceProposal,
    InvestmentThemesAnalysis,
    MissingAttributesAnalysis,
    NewEntitiesAnalysis,
)


# ---------------------------------------------------------------------------
# Pretty-printing individual analysis results
# ---------------------------------------------------------------------------


def _conf(item) -> str:
    c = getattr(item, "confidence", None)
    return c.value if isinstance(c, ConfidenceLevel) else str(c or "")


def print_investment_themes(data: InvestmentThemesAnalysis) -> None:
    print(f"\n{'='*60}")
    print("INVESTMENT THEMES ANALYSIS")
    print(f"{'='*60}")
    print(f"\nSummary: {data.summary[:300]}{'...' if len(data.summary) > 300 else ''}")
    print(f"\nThemes Found: {len(data.themes)}")
    for i, t in enumerate(data.themes, 1):
        print(f"\n  {i}. {t.name} [{_conf(t)}]")
        print(f"     {t.description[:100]}...")
        if t.market_size:
            print(f"     Market Size: {t.market_size}")
        if t.growth_projection:
            print(f"     Growth: {t.growth_projection}")
        if t.key_sectors:
            print(f"     Sectors: {', '.join(t.key_sectors[:5])}")
    if data.recommendations:
        print("\nRecommendations:")
        for rec in data.recommendations[:3]:
            print(f"  - {rec}")


def print_new_entities(data: NewEntitiesAnalysis) -> None:
    print(f"\n{'='*60}")
    print("NEW ENTITIES ANALYSIS")
    print(f"{'='*60}")
    print(f"\nSummary: {data.summary[:300]}{'...' if len(data.summary) > 300 else ''}")
    print(f"\nSuggested Nodes: {len(data.suggested_nodes)}")
    for i, n in enumerate(data.suggested_nodes, 1):
        print(f"\n  {i}. {n.label} [{_conf(n)}]")
        print(f"     Key: {n.key_property}")
        print(f"     {n.description[:100]}...")
        if n.properties:
            props = [f"{p.name}:{p.property_type}" for p in n.properties[:3]]
            print(f"     Properties: {', '.join(props)}")


def print_missing_attributes(data: MissingAttributesAnalysis) -> None:
    print(f"\n{'='*60}")
    print("MISSING ATTRIBUTES ANALYSIS")
    print(f"{'='*60}")
    print(f"\nSummary: {data.summary[:300]}{'...' if len(data.summary) > 300 else ''}")
    print(f"\nSuggested Attributes: {len(data.suggested_attributes)}")
    for i, a in enumerate(data.suggested_attributes, 1):
        print(f"\n  {i}. {a.target_label}.{a.property_name}: {a.property_type} [{_conf(a)}]")
        print(f"     {a.description[:100]}...")
        if a.example_values:
            print(f"     Examples: {', '.join(str(v) for v in a.example_values[:3])}")


def print_implied_relationships(data: ImpliedRelationshipsAnalysis) -> None:
    print(f"\n{'='*60}")
    print("IMPLIED RELATIONSHIPS ANALYSIS")
    print(f"{'='*60}")
    print(f"\nSummary: {data.summary[:300]}{'...' if len(data.summary) > 300 else ''}")
    print(f"\nSuggested Relationships: {len(data.suggested_relationships)}")
    for i, r in enumerate(data.suggested_relationships, 1):
        print(f"\n  {i}. ({r.source_label})-[{r.relationship_type}]->({r.target_label}) [{_conf(r)}]")
        print(f"     {r.description[:100]}...")
        if r.properties:
            props = [f"{p.name}:{p.property_type}" for p in r.properties[:3]]
            print(f"     Properties: {', '.join(props)}")


def print_analysis_result(result: AnalysisResult) -> None:
    """Print a single result with type-appropriate formatting."""
    if not result.success:
        print(f"\n[FAILED] {result.name}: {result.error}")
        return

    if result.reasoning:
        print(f"\n[Reasoning] {result.reasoning[:200]}...")

    d = result.data
    if isinstance(d, InvestmentThemesAnalysis):
        print_investment_themes(d)
    elif isinstance(d, NewEntitiesAnalysis):
        print_new_entities(d)
    elif isinstance(d, MissingAttributesAnalysis):
        print_missing_attributes(d)
    elif isinstance(d, ImpliedRelationshipsAnalysis):
        print_implied_relationships(d)


def print_response_summary(resp: AugmentationResponse) -> None:
    """Print a compact summary of the full augmentation response."""
    print("\n" + "=" * 70)
    print("AUGMENTATION ANALYSIS RESULTS")
    print("=" * 70)
    print(f"\n  Success:          {resp.success}")
    print(f"  Total suggestions: {resp.total_suggestions}")
    print(f"  High confidence:   {resp.high_confidence_count}")

    if resp.analysis.investment_themes:
        t = resp.analysis.investment_themes
        print(f"\n  --- Investment Themes ({len(t.themes)}) ---")
        for th in t.themes[:3]:
            print(f"    - {th.name} [{_conf(th)}]")

    if resp.all_suggested_nodes:
        print(f"\n  --- Suggested Nodes ({len(resp.all_suggested_nodes)}) ---")
        for n in resp.all_suggested_nodes[:5]:
            print(f"    - {n.label} [{_conf(n)}]")

    if resp.all_suggested_relationships:
        print(f"\n  --- Suggested Relationships ({len(resp.all_suggested_relationships)}) ---")
        for r in resp.all_suggested_relationships[:5]:
            print(f"    - ({r.source_label})-[{r.relationship_type}]->({r.target_label}) [{_conf(r)}]")

    if resp.all_suggested_attributes:
        print(f"\n  --- Suggested Attributes ({len(resp.all_suggested_attributes)}) ---")
        for a in resp.all_suggested_attributes[:5]:
            print(f"    - {a.target_label}.{a.property_name}: {a.property_type} [{_conf(a)}]")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Instance proposals and confidence filtering
# ---------------------------------------------------------------------------


def print_instance_proposal(proposal: InstanceProposal, index: int) -> None:
    """Print a single instance proposal."""
    src = proposal.source_node
    tgt = proposal.target_node
    print(
        f"\n  {index}. ({src.label} {src.key_value})"
        f"-[{proposal.relationship_type}]->"
        f"({tgt.label} {tgt.key_value})"
        f" [{_conf(proposal)}]"
    )
    print(f"     Source: {proposal.source_document}")
    print(f"     Evidence: \"{proposal.extracted_phrase[:120]}{'...' if len(proposal.extracted_phrase) > 120 else ''}\"")
    print(f"     Rationale: {proposal.rationale[:120]}{'...' if len(proposal.rationale) > 120 else ''}")


def print_filtered_proposals(filtered: FilteredProposals) -> None:
    """Print instance proposals grouped by confidence bucket."""
    total = len(filtered.auto_approve) + len(filtered.flagged) + len(filtered.review)

    print("\n" + "=" * 70)
    print("INSTANCE PROPOSALS — CONFIDENCE FILTERING")
    print("=" * 70)
    print(f"\n  Total proposals: {total}")
    print(f"  AUTO-APPROVE (HIGH):  {len(filtered.auto_approve)}")
    print(f"  FLAGGED (MEDIUM):     {len(filtered.flagged)}")
    print(f"  REVIEW (LOW):         {len(filtered.review)}")

    if filtered.auto_approve:
        print(f"\n{'─'*60}")
        print("AUTO-APPROVE — will write to Neo4j")
        print(f"{'─'*60}")
        for i, p in enumerate(filtered.auto_approve, 1):
            print_instance_proposal(p, i)

    if filtered.flagged:
        print(f"\n{'─'*60}")
        print("FLAGGED — approved with flag, logged for review")
        print(f"{'─'*60}")
        for i, p in enumerate(filtered.flagged, 1):
            print_instance_proposal(p, i)

    if filtered.review:
        print(f"\n{'─'*60}")
        print("REVIEW — queued, not written")
        print(f"{'─'*60}")
        for i, p in enumerate(filtered.review, 1):
            print_instance_proposal(p, i)

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# PASS/FAIL test harness
# ---------------------------------------------------------------------------


class ValidationHarness:
    """Collects PASS/FAIL results and prints a summary table."""

    def __init__(self) -> None:
        self._results: list[tuple[str, bool, str]] = []

    def record(self, name: str, passed: bool, detail: str = "") -> None:
        self._results.append((name, passed, detail))
        tag = "PASS" if passed else "FAIL"
        line = f"  [{tag}] {name}"
        if detail:
            line += f" — {detail}"
        print(line)

    @property
    def all_passed(self) -> bool:
        return all(p for _, p, _ in self._results)

    def print_summary(self) -> None:
        passed = sum(1 for _, p, _ in self._results if p)
        failed = sum(1 for _, p, _ in self._results if not p)
        total = len(self._results)

        print("\n" + "=" * 60)
        print(f"Results: {passed} passed, {failed} failed, {total} total")
        print("=" * 60)
        for name, p, detail in self._results:
            tag = "PASS" if p else "FAIL"
            line = f"  [{tag}] {name}"
            if detail:
                line += f" — {detail}"
            print(line)
        print("=" * 60)
        if failed == 0:
            print("SUCCESS: All checks passed")
        else:
            print(f"FAILURE: {failed} check(s) failed")
        print("=" * 60)
