"""DSPy analyzer modules for graph augmentation.

Each analyzer wraps a ``dspy.ChainOfThought`` predictor with a typed
signature and returns a uniform ``AnalysisResult`` dataclass.

The composite ``GraphAugmentationAnalyzer`` uses ``dspy.Parallel`` to
run all four analyses concurrently.  ``dspy.Parallel`` properly
propagates DSPy's thread-local settings (configured LM, adapter, etc.)
to worker threads, which a plain ``ThreadPoolExecutor`` would not.

References:
    https://dspy.ai/learn/programming/modules/
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import dspy

from semantic_auth.schemas import (
    AugmentationAnalysis,
    AugmentationResponse,
    ImpliedRelationshipsAnalysis,
    InvestmentThemesAnalysis,
    MissingAttributesAnalysis,
    NewEntitiesAnalysis,
    SuggestedAttribute,
    SuggestedNode,
    SuggestedRelationship,
)
from semantic_auth.signatures import (
    ImpliedRelationshipsSignature,
    InvestmentThemesSignature,
    MissingAttributesSignature,
    NewEntitiesSignature,
)


# ---------------------------------------------------------------------------
# Unified result envelope
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AnalysisResult:
    """Return type shared by every analyzer."""

    name: str
    success: bool
    data: Any = None
    error: str | None = None
    reasoning: str | None = None


# ---------------------------------------------------------------------------
# Individual analyzers (one per signature)
# ---------------------------------------------------------------------------


class InvestmentThemesAnalyzer(dspy.Module):
    """Extract investment themes from market research documents."""

    def __init__(self) -> None:
        super().__init__()
        self.analyze = dspy.ChainOfThought(InvestmentThemesSignature)

    def forward(self, document_context: str) -> AnalysisResult:
        try:
            result = self.analyze(document_context=document_context)
            return AnalysisResult(
                name="investment_themes",
                success=True,
                data=result.analysis,
                reasoning=getattr(result, "reasoning", None),
            )
        except Exception as e:
            return AnalysisResult(name="investment_themes", success=False, error=str(e))


class NewEntitiesAnalyzer(dspy.Module):
    """Suggest new node types from document analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.analyze = dspy.ChainOfThought(NewEntitiesSignature)

    def forward(self, document_context: str) -> AnalysisResult:
        try:
            result = self.analyze(document_context=document_context)
            return AnalysisResult(
                name="new_entities",
                success=True,
                data=result.analysis,
                reasoning=getattr(result, "reasoning", None),
            )
        except Exception as e:
            return AnalysisResult(name="new_entities", success=False, error=str(e))


class MissingAttributesAnalyzer(dspy.Module):
    """Identify attributes missing from existing graph nodes."""

    def __init__(self) -> None:
        super().__init__()
        self.analyze = dspy.ChainOfThought(MissingAttributesSignature)

    def forward(self, document_context: str) -> AnalysisResult:
        try:
            result = self.analyze(document_context=document_context)
            return AnalysisResult(
                name="missing_attributes",
                success=True,
                data=result.analysis,
                reasoning=getattr(result, "reasoning", None),
            )
        except Exception as e:
            return AnalysisResult(name="missing_attributes", success=False, error=str(e))


class ImpliedRelationshipsAnalyzer(dspy.Module):
    """Discover relationships implied but not captured in the graph."""

    def __init__(self) -> None:
        super().__init__()
        self.analyze = dspy.ChainOfThought(ImpliedRelationshipsSignature)

    def forward(self, document_context: str) -> AnalysisResult:
        try:
            result = self.analyze(document_context=document_context)
            return AnalysisResult(
                name="implied_relationships",
                success=True,
                data=result.analysis,
                reasoning=getattr(result, "reasoning", None),
            )
        except Exception as e:
            return AnalysisResult(
                name="implied_relationships", success=False, error=str(e)
            )


# ---------------------------------------------------------------------------
# Composite orchestrator
# ---------------------------------------------------------------------------

#: Canonical name -> analyzer class mapping.
ANALYZER_REGISTRY: dict[str, type[dspy.Module]] = {
    "investment_themes": InvestmentThemesAnalyzer,
    "new_entities": NewEntitiesAnalyzer,
    "missing_attributes": MissingAttributesAnalyzer,
    "implied_relationships": ImpliedRelationshipsAnalyzer,
}


class GraphAugmentationAnalyzer(dspy.Module):
    """Run multiple analyzers and consolidate into an ``AugmentationResponse``.

    Uses ``dspy.Parallel`` so all requested analyses execute concurrently.
    """

    def __init__(self, analyses: list[str] | None = None) -> None:
        super().__init__()
        names = analyses or list(ANALYZER_REGISTRY)
        self._names = [n for n in names if n in ANALYZER_REGISTRY]
        # Register sub-modules so DSPy can track parameters / save state.
        for name in self._names:
            setattr(self, name, ANALYZER_REGISTRY[name]())

    def forward(self, document_context: str) -> AugmentationResponse:
        example = dspy.Example(
            document_context=document_context,
        ).with_inputs("document_context")

        exec_pairs = [(getattr(self, n), example) for n in self._names]

        print(f"\n  Running {len(exec_pairs)} analyses via dspy.Parallel ...")
        t0 = time.time()

        parallel = dspy.Parallel(
            num_threads=len(exec_pairs),
            max_errors=len(exec_pairs),
            provide_traceback=True,
        )
        raw_results: list[AnalysisResult] = parallel(exec_pairs)

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s\n")

        return self._consolidate(raw_results)

    def run_single(self, name: str, document_context: str) -> AnalysisResult:
        """Run one analysis by name."""
        if name not in self._names:
            raise ValueError(
                f"Unknown analysis: {name!r}. Choose from {self._names}"
            )
        return getattr(self, name)(document_context)

    # -- private -----------------------------------------------------------

    @staticmethod
    def _consolidate(results: list[AnalysisResult]) -> AugmentationResponse:
        analysis = AugmentationAnalysis()
        nodes: list[SuggestedNode] = []
        rels: list[SuggestedRelationship] = []
        attrs: list[SuggestedAttribute] = []
        any_ok = False

        for r in results:
            if r is None:
                print("  [unknown] FAILED: None")
                continue
            if not r.success or r.data is None:
                print(f"  [{r.name}] FAILED: {r.error or 'no data'}")
                continue

            any_ok = True
            print(f"  [{r.name}] OK")

            if isinstance(r.data, InvestmentThemesAnalysis):
                analysis.investment_themes = r.data
            elif isinstance(r.data, NewEntitiesAnalysis):
                analysis.new_entities = r.data
                nodes.extend(r.data.suggested_nodes)
            elif isinstance(r.data, MissingAttributesAnalysis):
                analysis.missing_attributes = r.data
                attrs.extend(r.data.suggested_attributes)
            elif isinstance(r.data, ImpliedRelationshipsAnalysis):
                analysis.implied_relationships = r.data
                rels.extend(r.data.suggested_relationships)

        resp = AugmentationResponse(
            success=any_ok,
            analysis=analysis,
            all_suggested_nodes=nodes,
            all_suggested_relationships=rels,
            all_suggested_attributes=attrs,
        )
        resp.compute_statistics()
        return resp
