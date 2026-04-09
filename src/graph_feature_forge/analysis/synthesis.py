"""LLM synthesis for gap analysis.

Each query type assembles structured data context (from
:class:`StructuredDataAccess`) and retrieved document chunks (from
:class:`DocumentRetrieval`) into a prompt, sends it to the foundation
model endpoint, and returns the gap analysis text.

Five query types are supported:

=========================  ============================  =========================
Query type                 Structured data method        Document retrieval focus
=========================  ============================  =========================
interest_holding_gaps      get_portfolio_holdings()       customer profiles + themes
risk_alignment             get_customer_profiles()        customer profiles + risk
data_quality_gaps          get_data_completeness()        customer profiles
investment_themes          (none — doc-only)              market research + themes
comprehensive              get_all_structured_context()   all documents
=========================  ============================  =========================

The ``fetch_gap_analysis()`` convenience function runs a comprehensive
analysis and feeds the result into ``GraphAugmentationAnalyzer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from graph_feature_forge.data.retrieval import DocumentRetrieval, Neo4jRetrieval
from graph_feature_forge.data.structured_data import StructuredDataAccess

# Type alias: takes a list of message dicts, returns response text.
LLMCaller = Callable[[list[dict[str, str]]], str]

# Either retrieval backend — both expose query() and format_context().
DocumentRetrievalLike = DocumentRetrieval | Neo4jRetrieval


# ---------------------------------------------------------------------------
# Gap analysis query prompts
# ---------------------------------------------------------------------------

INTEREST_HOLDING_GAP_INSTRUCTIONS = """\
Analyze customer investment interests vs their actual portfolio holdings.

For each customer with profile documents:
1. What investment interests have they expressed? (renewable energy, ESG,
   technology, real estate, fintech, etc.)
2. What stocks/sectors do they currently hold in their portfolios?
3. Identify gaps where expressed interests don't match holdings.

Format your response as a detailed analysis for each customer including:
- Customer ID and name
- Expressed interests (from profile documents)
- Current holdings (from portfolio data)
- Identified gaps (interests not reflected in portfolio)
- Specific quotes from their profile showing the interest

Focus on customers whose profiles contain stated interests, financial goals,
or sector preferences that diverge from their current portfolio composition."""

RISK_ALIGNMENT_INSTRUCTIONS = """\
Analyze risk profile alignment between customer profiles and portfolios.

For each customer:
1. What is their stated risk tolerance in the structured database?
2. What risk-related language appears in their profile documents?
3. Does their actual portfolio composition match their risk profile?

Identify customers where:
- Profile narrative suggests different risk tolerance than database field
- Portfolio composition doesn't match stated risk tolerance
- Risk preferences have evolved based on profile updates

Include specific evidence from both structured data and profile documents."""

DATA_QUALITY_GAP_INSTRUCTIONS = """\
Identify data quality gaps between customer profiles and structured database.

Compare the information in customer profile documents against what's stored
in the structured customer database tables.

Look for:
1. Personal attributes mentioned in profiles but missing from database
   (occupation details, employer, life stage, family situation)
2. Financial goals mentioned in profiles but not captured as structured data
3. Investment preferences detailed in profiles but not in database fields
4. Contact preferences or communication style notes
5. Any temporal information (retirement timeline, education savings goals)

For each gap found, provide:
- The customer ID
- The attribute/information found in the profile
- Whether it exists in the structured database
- The exact quote or reference from the profile document"""

INVESTMENT_THEMES_INSTRUCTIONS = """\
Extract investment themes from market research documents.

Analyze all market research, sector analysis, and investment guide documents to identify:

1. Major investment themes being discussed
   - Theme name (e.g., "Renewable Energy Transition", "AI Infrastructure")
   - Market size and growth projections mentioned
   - Key sectors within the theme
   - Specific companies mentioned as opportunities

2. Sector-specific insights
   - Technology sector trends and key players
   - Renewable energy opportunities and companies
   - Financial sector analysis
   - Any emerging sectors discussed

3. Risk considerations mentioned
   - Valuation concerns
   - Market timing considerations
   - Sector-specific risks

For each theme, include the source document and relevant quotes."""

COMPREHENSIVE_INSTRUCTIONS = """\
Perform comprehensive gap analysis for graph augmentation opportunities.

This analysis will identify information in documents that should be captured
as new nodes, relationships, or attributes in the Neo4j graph.

PART 1: Customer Interest-Holding Gaps
For each customer with a profile document:
- What investment interests are expressed in their profiles?
- What do they currently hold in their portfolios?
- What's the gap between interests and holdings?
- Quote the specific profile text showing their interests.

PART 2: Missing Entity Relationships
What relationships are implied in documents but not in the graph?
- Customer-to-interest relationships (INTERESTED_IN)
- Customer-to-goal relationships (HAS_GOAL)
- Customer-to-employer relationships (WORKS_AT)
- Customer similarity patterns (SIMILAR_TO)

PART 3: Missing Customer Attributes
What customer attributes appear in profiles but aren't in structured data?
- Occupation and employer details
- Life stage (mid-career, approaching retirement, etc.)
- Investment philosophy
- Communication preferences

PART 4: Investment Theme Entities
What investment themes from research should become graph nodes?
- Theme names and descriptions
- Associated sectors and companies
- Market size and growth data

Provide specific evidence and quotes for each finding."""


# ---------------------------------------------------------------------------
# Document retrieval queries (tuned for each analysis type)
# ---------------------------------------------------------------------------

_RETRIEVAL_QUERIES: dict[str, str] = {
    "interest_holding_gaps": (
        "Customer investment interests and preferences. "
        "Renewable energy, ESG, sustainable investing, technology, "
        "fintech, capital preservation, dividend income, growth."
    ),
    "risk_alignment": (
        "Customer risk tolerance and risk profile. "
        "Conservative, moderate, aggressive investment strategy. "
        "Portfolio risk alignment and preferences."
    ),
    "data_quality_gaps": (
        "Customer personal details, occupation, employer, "
        "financial goals, investment philosophy, life stage, "
        "communication preferences."
    ),
    "investment_themes": (
        "Investment themes, market trends, sector analysis. "
        "Renewable energy transition, technology sector, "
        "market size, growth projections, emerging opportunities."
    ),
    "comprehensive": (
        "Customer investment interests, risk profiles, "
        "personal attributes, market research themes, "
        "portfolio gaps, graph augmentation opportunities."
    ),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GapAnalysisResult:
    """Result from a gap analysis synthesis call."""

    query_type: str
    response: str
    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a financial data analyst specializing in graph-based knowledge \
representation. You are given structured portfolio and customer data from \
a Neo4j graph database alongside excerpts from customer profile documents \
and market research reports. Your task is to analyze this data to identify \
gaps and enrichment opportunities for the graph.

Be specific and evidence-based. When you identify a gap or finding, cite \
the structured data field or quote the document excerpt that supports it. \
Use customer IDs (e.g., C0001) when referencing customers."""


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class GapAnalysisSynthesizer:
    """Synthesize gap analysis by combining structured data and documents.

    Each method corresponds to one of the five gap analysis query types
    and returns a :class:`GapAnalysisResult`.

    Args:
        structured_data: A configured :class:`StructuredDataAccess`
            instance for querying the Delta tables.
        retrieval: A configured :class:`DocumentRetrieval` or
            :class:`Neo4jRetrieval` instance for similarity-searching
            the document chunks.
        llm_caller: Callable that takes a list of message dicts
            (``[{"role": ..., "content": ...}]``) and returns the
            model's response text.  Use :func:`make_sdk_caller` to
            create one.
        retrieval_top_k: Number of document chunks to retrieve per
            query (default 5).
    """

    def __init__(
        self,
        structured_data: StructuredDataAccess,
        retrieval: DocumentRetrievalLike,
        llm_caller: LLMCaller,
        retrieval_top_k: int = 5,
        enrichment_context: str | None = None,
    ) -> None:
        self._data = structured_data
        self._retrieval = retrieval
        self._llm = llm_caller
        self._top_k = retrieval_top_k
        self._enrichment_context = enrichment_context

    def _synthesize(
        self,
        query_type: str,
        instructions: str,
        structured_context: str | None,
    ) -> GapAnalysisResult:
        """Run a single synthesis call.

        Retrieves documents using the pre-tuned query for
        *query_type*, assembles the prompt, and calls the LLM.
        """
        retrieval_query = _RETRIEVAL_QUERIES[query_type]
        doc_context = self._retrieval.format_context(
            retrieval_query, top_k=self._top_k
        )

        # Build user message with clearly labeled sections
        sections: list[str] = []

        if structured_context:
            sections.append(
                "## Structured Data (Portfolio / Account / Customer Data)\n\n"
                + structured_context
            )

        sections.append(
            "## Document Excerpts (Customer Profiles, Market Research)\n\n"
            + doc_context
        )

        if self._enrichment_context:
            sections.append(self._enrichment_context)

        sections.append("## Analysis Instructions\n\n" + instructions)

        user_message = "\n\n---\n\n".join(sections)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            response = self._llm(messages)
            return GapAnalysisResult(
                query_type=query_type,
                response=response,
                success=True,
            )
        except Exception as exc:
            return GapAnalysisResult(
                query_type=query_type,
                response="",
                success=False,
                error=str(exc),
            )

    # -----------------------------------------------------------------
    # Per-query-type methods
    # -----------------------------------------------------------------

    def analyze_interest_holding_gaps(self) -> GapAnalysisResult:
        """Compare customer-stated interests against portfolio holdings."""
        return self._synthesize(
            "interest_holding_gaps",
            INTEREST_HOLDING_GAP_INSTRUCTIONS,
            self._data.get_portfolio_holdings(),
        )

    def analyze_risk_alignment(self) -> GapAnalysisResult:
        """Check whether portfolio composition matches risk profiles."""
        return self._synthesize(
            "risk_alignment",
            RISK_ALIGNMENT_INSTRUCTIONS,
            self._data.get_customer_profiles(),
        )

    def analyze_data_quality_gaps(self) -> GapAnalysisResult:
        """Find profile attributes missing from the structured database."""
        return self._synthesize(
            "data_quality_gaps",
            DATA_QUALITY_GAP_INSTRUCTIONS,
            self._data.get_data_completeness(),
        )

    def extract_investment_themes(self) -> GapAnalysisResult:
        """Extract investment themes from market research documents."""
        return self._synthesize(
            "investment_themes",
            INVESTMENT_THEMES_INSTRUCTIONS,
            None,  # doc-only — no structured data needed
        )

    def run_comprehensive_analysis(self) -> GapAnalysisResult:
        """Run all four analyses in a single comprehensive prompt."""
        return self._synthesize(
            "comprehensive",
            COMPREHENSIVE_INSTRUCTIONS,
            self._data.get_all_structured_context(),
        )


# ---------------------------------------------------------------------------
# LLM caller factory
# ---------------------------------------------------------------------------


def make_sdk_caller(
    endpoint: str = "databricks-claude-sonnet-4-6",
    max_tokens: int = 4096,
) -> LLMCaller:
    """Create an LLM caller using the Databricks SDK.

    Uses ``WorkspaceClient.serving_endpoints.query()`` with
    ``ChatMessage`` / ``ChatMessageRole`` objects (required by the
    SDK version on Databricks serverless runtimes).  Callers pass
    plain dicts — the factory converts them internally.

    Works both locally (``DATABRICKS_HOST``/``DATABRICKS_TOKEN``)
    and on-cluster (automatic runtime credentials).

    Args:
        endpoint: Databricks foundation model endpoint name.
        max_tokens: Maximum tokens in the response.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    _role_map = {
        "system": ChatMessageRole.SYSTEM,
        "user": ChatMessageRole.USER,
        "assistant": ChatMessageRole.ASSISTANT,
    }

    from databricks.sdk.config import Config

    cfg = Config()
    cfg.http_timeout_seconds = 600  # 10 min — synthesis can take 4+ min
    wc = WorkspaceClient(config=cfg)

    def call(messages: list[dict[str, str]]) -> str:
        sdk_messages = [
            ChatMessage(
                role=_role_map[m["role"]],
                content=m["content"],
            )
            for m in messages
        ]
        response = wc.serving_endpoints.query(
            name=endpoint,
            messages=sdk_messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    return call


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def fetch_gap_analysis(
    structured_data: StructuredDataAccess,
    retrieval: DocumentRetrievalLike,
    llm_caller: LLMCaller,
    enrichment_context: str | None = None,
) -> str:
    """Run comprehensive gap analysis and return the response text.

    The pipeline entry point calls this and feeds the result into
    ``GraphAugmentationAnalyzer``.

    Args:
        enrichment_context: Optional prior-enrichment summary to include
            in the prompt so the LLM avoids re-proposing existing
            relationships.

    Raises:
        RuntimeError: If the synthesis call fails.
    """
    synth = GapAnalysisSynthesizer(
        structured_data, retrieval, llm_caller,
        enrichment_context=enrichment_context,
    )
    result = synth.run_comprehensive_analysis()

    if not result.success:
        raise RuntimeError(f"Gap analysis failed: {result.error}")

    return result.response
