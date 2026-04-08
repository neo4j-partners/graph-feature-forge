"""Generate HTML documents from CSV entity data.

In live mode, calls the Databricks LLM endpoint via the OpenAI-compatible
SDK to produce realistic narrative HTML.  In dry-run mode, fills simple
templates with entity data so the full pipeline can be tested without
network access.
"""

from __future__ import annotations

import csv
import logging
import random
import re
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import GeneratorConfig
from .models import EntityReferences, GeneratedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> List[Dict[str, str]]:
    """Read a CSV into a list of dicts.  Returns empty list if missing."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_entities(cfg: GeneratorConfig) -> Dict[str, List[Dict[str, str]]]:
    """Load all CSV entity files and return a dict keyed by entity type."""
    csv_dir = cfg.csv_dir
    return {
        "customers": _load_csv(csv_dir / "customers.csv"),
        "companies": _load_csv(csv_dir / "companies.csv"),
        "banks": _load_csv(csv_dir / "banks.csv"),
        "accounts": _load_csv(csv_dir / "accounts.csv"),
        "stocks": _load_csv(csv_dir / "stocks.csv"),
        "portfolio_holdings": _load_csv(csv_dir / "portfolio_holdings.csv"),
        "transactions": _load_csv(csv_dir / "transactions.csv"),
    }


# ---------------------------------------------------------------------------
# Entity selectors — pick "interesting" rows for documents
# ---------------------------------------------------------------------------


def _select_customers(
    entities: Dict[str, List[Dict[str, str]]],
    count: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """Pick customers with varied risk profiles and incomes."""
    customers = entities["customers"]
    if not customers:
        return []

    # Separate labeled vs unlabeled, prefer labeled for profiles
    labeled = [c for c in customers if c.get("risk_profile")]
    unlabeled = [c for c in customers if not c.get("risk_profile")]

    # Group by risk profile
    by_risk: Dict[str, List[Dict[str, str]]] = {}
    for c in labeled:
        by_risk.setdefault(c["risk_profile"], []).append(c)

    selected: List[Dict[str, str]] = []
    # Take at least one from each risk class
    for profile_group in by_risk.values():
        if profile_group:
            selected.append(rng.choice(profile_group))

    # Fill remaining from all labeled, preferring diverse incomes
    remaining = [c for c in labeled if c not in selected]
    remaining.sort(key=lambda c: float(c.get("annual_income", 0)))
    # Take from top, bottom, and middle of income range
    if remaining and len(selected) < count:
        step = max(1, len(remaining) // (count - len(selected)))
        for i in range(0, len(remaining), step):
            if len(selected) >= count:
                break
            if remaining[i] not in selected:
                selected.append(remaining[i])

    # If still short, add unlabeled
    if len(selected) < count and unlabeled:
        rng.shuffle(unlabeled)
        selected.extend(unlabeled[: count - len(selected)])

    return selected[:count]


def _select_companies(
    entities: Dict[str, List[Dict[str, str]]],
    count: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """Pick companies across diverse sectors."""
    companies = entities["companies"]
    if not companies:
        return []

    by_sector: Dict[str, List[Dict[str, str]]] = {}
    for c in companies:
        by_sector.setdefault(c.get("sector", "Unknown"), []).append(c)

    selected: List[Dict[str, str]] = []
    # One from each sector first
    for sector_group in by_sector.values():
        if sector_group and len(selected) < count:
            selected.append(rng.choice(sector_group))

    # Fill remaining randomly
    remaining = [c for c in companies if c not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)])

    return selected[:count]


def _select_banks(
    entities: Dict[str, List[Dict[str, str]]],
    count: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """Pick banks with diverse types and sizes."""
    banks = entities["banks"]
    if not banks:
        return []
    rng.shuffle(banks)
    return banks[:count]


# ---------------------------------------------------------------------------
# Sector grouping for market analysis docs
# ---------------------------------------------------------------------------

_SECTOR_TOPICS = [
    "Technology",
    "Financial Services",
    "Healthcare",
    "Energy",
    "Consumer Goods",
    "Industrials",
    "Real Estate",
    "Telecommunications",
    "Utilities",
    "Materials",
]

_INVESTMENT_GUIDE_TOPICS = [
    ("Portfolio Diversification Strategies", "investment_guide"),
    ("Aggressive Growth Investment Playbook", "investment_guide"),
    ("Conservative Income-Focused Investing", "investment_guide"),
    ("ESG and Sustainable Investing Guide", "investment_guide"),
    ("Tax-Efficient Investment Strategies", "investment_guide"),
    ("Retirement Planning for Young Professionals", "financial_planning"),
    ("Estate Planning and Wealth Transfer Guide", "financial_planning"),
    ("Emergency Fund and Cash Management", "financial_planning"),
    ("International Investing Guide", "investment_guide"),
    ("Fixed Income and Bond Investment Strategies", "investment_guide"),
]

_REGULATORY_TOPICS = [
    ("Banking Regulatory Compliance Update 2024", "regulatory"),
    ("Anti-Money Laundering Best Practices", "regulatory"),
    ("Consumer Data Privacy in Financial Services", "regulatory"),
]


# ---------------------------------------------------------------------------
# Behavioral signal generation for discoverable customer preferences
# ---------------------------------------------------------------------------

# Sector weights for scoring portfolio style (mirrors csv_generator bias).
# Higher weight = stronger signal toward that style.
_STYLE_SECTOR_SCORES: Dict[str, Dict[str, float]] = {
    "growth": {
        "Technology": 5.0,
        "Healthcare": 4.0,
        "Communication Services": 3.5,
        "Consumer Discretionary": 2.5,
    },
    "income": {
        "Utilities": 5.0,
        "Consumer Staples": 4.5,
        "Real Estate": 4.0,
        "Materials": 2.5,
        "Energy": 2.5,
    },
    "balanced": {
        "Financial Services": 4.0,
        "Industrials": 3.5,
        "Consumer Discretionary": 3.0,
        "Healthcare": 2.5,
    },
}

# Behavioral signals the enrichment loop can discover.  Phrased as
# advisor observations so they read naturally in profile documents.
_ADVISOR_OBSERVATIONS: Dict[str, List[str]] = {
    "growth": [
        "actively monitors market trends and frequently discusses portfolio positioning",
        "has expressed comfort with short-term volatility and willingness to accept higher risk for greater returns",
        "regularly inquires about emerging sectors and disruptive technology companies",
        "prefers concentrated positions in high-conviction ideas rather than broad diversification",
        "has discussed momentum-based strategies and growth stock screening criteria",
        "reviews portfolio performance weekly and stays current with market news",
        "has expressed interest in options strategies to amplify upside exposure",
        "has indicated willingness to reduce dividend income in favor of capital appreciation",
    ],
    "income": [
        "prioritizes capital preservation and consistent income generation from investments",
        "has expressed strong preference for dividend-paying stocks and fixed-income instruments",
        "prefers gradual scheduled portfolio adjustments over reactive trading",
        "has discussed estate planning and intergenerational wealth transfer strategies",
        "values portfolio stability and tends to avoid speculative or high-volatility positions",
        "focuses on downside protection and has inquired about hedging strategies",
        "has expressed interest in bond laddering and systematic withdrawal strategies",
        "prefers companies with long track records of increasing dividend payments",
    ],
    "balanced": [
        "seeks a balance between growth opportunities and downside protection",
        "has expressed interest in diversified sector exposure through both individual stocks and index funds",
        "reviews portfolio quarterly with a focus on risk-adjusted returns",
        "has explored both growth-oriented and income-generating investment vehicles",
        "appreciates a structured rebalancing approach tied to target allocations",
        "remains open to tactical sector shifts within a broadly diversified framework",
        "has expressed interest in sustainable investing and ESG-screened portfolios",
        "has discussed dollar-cost averaging into new positions over time",
    ],
}

_INVESTMENT_GOALS: Dict[str, List[str]] = {
    "growth": [
        "building wealth aggressively for early retirement",
        "maximizing long-term capital appreciation over the next decade",
        "accumulating capital to fund future entrepreneurial ventures",
        "growing a concentrated portfolio of high-conviction positions",
    ],
    "income": [
        "generating reliable retirement income to supplement Social Security",
        "preserving wealth for transfer to the next generation",
        "funding education expenses for dependents through investment returns",
        "building a stable income stream that keeps pace with inflation",
    ],
    "balanced": [
        "achieving long-term financial independence with moderate risk",
        "building a diversified retirement portfolio with growth and income",
        "saving for a major life milestone while maintaining a safety net",
        "funding both near-term spending needs and long-term wealth goals",
    ],
}

_SECTOR_INTERESTS: Dict[str, List[str]] = {
    "growth": [
        "artificial intelligence and machine learning applications",
        "cloud computing and enterprise software platforms",
        "biotechnology and gene therapy breakthroughs",
        "electric vehicles and clean energy technology",
        "digital payments and fintech innovation",
        "cybersecurity and data protection companies",
        "space exploration and satellite technology",
    ],
    "income": [
        "regulated utilities and essential infrastructure",
        "real estate investment trusts with stable occupancy",
        "consumer staples companies with pricing power",
        "municipal bonds and investment-grade fixed income",
        "dividend growth strategies with 20-year track records",
        "pipeline and energy infrastructure master limited partnerships",
        "healthcare REITs and senior living facilities",
    ],
    "balanced": [
        "sustainable investing and ESG-focused funds",
        "industrial automation and robotics",
        "healthcare innovation and medical devices",
        "renewable energy infrastructure and green bonds",
        "financial services and insurance companies",
        "broad market index funds with low expense ratios",
        "international diversification across developed markets",
    ],
}

_STYLE_LABELS: Dict[str, str] = {
    "growth": "growth-oriented",
    "income": "income-focused",
    "balanced": "balanced",
}

_STYLE_ELABORATIONS: Dict[str, str] = {
    "growth": (
        "capital appreciation and exposure to high-growth sectors, "
        "with a higher tolerance for short-term price fluctuations"
    ),
    "income": (
        "capital preservation, consistent dividend income, and "
        "stable investments with lower volatility"
    ),
    "balanced": (
        "diversified exposure across growth and value sectors, "
        "with regular rebalancing to manage risk"
    ),
}


def _derive_investment_style(
    holdings: List[Dict[str, str]],
    stocks: Dict[str, Dict[str, str]],
    companies: Dict[str, Dict[str, str]],
    customer: Dict[str, str],
) -> str:
    """Derive investment style from portfolio sector composition.

    Scores the customer's holdings against sector weights for each style.
    Falls back to income/credit heuristics when no holdings exist.
    Returns ``'growth'``, ``'income'``, or ``'balanced'``.
    """
    if holdings:
        scores: Dict[str, float] = {"growth": 0.0, "income": 0.0, "balanced": 0.0}
        for h in holdings:
            sid = h.get("stock_id", "")
            s = stocks.get(sid, {})
            co = companies.get(s.get("company_id", ""), {})
            sector = co.get("sector", "")
            for style, sector_weights in _STYLE_SECTOR_SCORES.items():
                scores[style] += sector_weights.get(sector, 1.0)
        return max(scores, key=scores.__getitem__)

    # Fallback: no holdings — use income and credit score heuristics
    try:
        income = float(customer.get("annual_income", 0))
        credit = int(customer.get("credit_score", 700))
    except (ValueError, TypeError):
        return "balanced"

    if income > 100_000 and credit < 700:
        return "growth"
    elif credit > 740:
        return "income"
    return "balanced"


def _build_customer_context(
    rng: random.Random,
    customer: Dict[str, str],
    holdings: List[Dict[str, str]],
    stocks: Dict[str, Dict[str, str]],
    companies: Dict[str, Dict[str, str]],
) -> Dict[str, object]:
    """Build behavioral context for a customer profile document.

    Derives investment style from portfolio composition and selects
    behavioral signals, goals, and interests that the enrichment loop
    can discover.
    """
    style = _derive_investment_style(holdings, stocks, companies, customer)

    # Top sectors from actual holdings
    sector_counts: Dict[str, int] = {}
    for h in holdings:
        sid = h.get("stock_id", "")
        s = stocks.get(sid, {})
        co = companies.get(s.get("company_id", ""), {})
        sector = co.get("sector", "")
        if sector:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    top_sectors = [
        s for s, _ in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
    ][:3]

    observations = rng.sample(_ADVISOR_OBSERVATIONS[style], 2)
    goal = rng.choice(_INVESTMENT_GOALS[style])
    interests = rng.sample(_SECTOR_INTERESTS[style], 2)

    return {
        "style": style,
        "style_label": _STYLE_LABELS[style],
        "style_elaboration": _STYLE_ELABORATIONS[style],
        "top_sectors": top_sectors,
        "observations": observations,
        "goal": goal,
        "interests": interests,
    }


# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------


def _customer_profile_prompt(
    customer: Dict[str, str],
    accounts: List[Dict[str, str]],
    holdings: List[Dict[str, str]],
    stocks: Dict[str, Dict[str, str]],
    companies: Dict[str, Dict[str, str]],
    bank_names: Dict[str, str],
    context: Dict[str, object],
) -> str:
    name = f"{customer['first_name']} {customer['last_name']}"
    cid = customer["customer_id"]
    risk = customer.get("risk_profile") or "not assessed"
    income = customer.get("annual_income", "unknown")
    credit = customer.get("credit_score", "unknown")
    emp = customer.get("employment_status", "unknown")
    city = customer.get("city", "")
    state = customer.get("state", "")
    dob = customer.get("date_of_birth", "")
    reg_date = customer.get("registration_date", "")

    acct_lines = []
    for a in accounts:
        bname = bank_names.get(a.get("bank_id", ""), "their bank")
        acct_lines.append(
            f"  - {a['account_type']} account at {bname} "
            f"(balance: ${a.get('balance', '?')})"
        )

    holding_lines = []
    for h in holdings:
        sid = h.get("stock_id", "")
        s = stocks.get(sid, {})
        co = companies.get(s.get("company_id", ""), {})
        ticker = s.get("ticker", "???")
        co_name = co.get("name", "Unknown Company")
        shares = h.get("shares", "?")
        holding_lines.append(f"  - {shares} shares of {co_name} ({ticker})")

    return textwrap.dedent(f"""\
        Write a detailed customer profile HTML document for a financial institution's
        knowledge base.  The document should be in the style of a wealth management
        client profile, 4-6 paragraphs of narrative prose.

        Use this exact HTML structure (no CSS, no scripts, plain semantic HTML):
        <!DOCTYPE html>
        <html>
        <head><title>Customer Profile - {name}</title></head>
        <body>
        <h1>Customer Profile: {name}</h1>
        <p>...</p>
        ...
        </body>
        </html>

        Facts to weave into the narrative:
        - Full name: {name}
        - Customer ID: {cid}
        - Location: {city}, {state}
        - Date of birth: {dob}
        - Registration date: {reg_date}
        - Employment status: {emp}
        - Annual income: ${income}
        - Credit score: {credit}
        - Risk profile: {risk}
        - Accounts:
        {chr(10).join(acct_lines) if acct_lines else "  (no accounts listed)"}
        - Portfolio holdings:
        {chr(10).join(holding_lines) if holding_lines else "  (no holdings listed)"}

        Advisor observations (weave naturally into the narrative — do NOT
        present as a bullet list or separate section):
        - {context['observations'][0]}
        - {context['observations'][1]}
        - Primary financial goal: {context['goal']}
        - Has expressed interest in: {context['interests'][0]} and {context['interests'][1]}
        - Portfolio concentration in: {', '.join(context['top_sectors']) if context['top_sectors'] else 'multiple'} sectors

        Write naturally, as if a financial advisor were summarizing the client.
        Weave the behavioral observations and interests into the narrative as
        things the advisor has noted during client meetings and portfolio reviews.
        They should feel like natural parts of the client description, not a
        separate data section.
        Reference the company names and ticker symbols of holdings.
        Do NOT include any markdown fences or commentary — output only the HTML.
    """)


def _company_analysis_prompt(
    company: Dict[str, str],
    stock: Optional[Dict[str, str]],
) -> str:
    name = company.get("name", "Unknown")
    ticker = company.get("ticker_symbol", "???")
    industry = company.get("industry", "")
    sector = company.get("sector", "")
    mcap = company.get("market_cap_billions", "")
    hq = company.get("headquarters", "")
    founded = company.get("founded_year", "")
    ceo = company.get("ceo", "")
    employees = company.get("employee_count", "")
    revenue = company.get("annual_revenue_billions", "")

    stock_info = ""
    if stock:
        stock_info = textwrap.dedent(f"""\
            Stock data:
            - Current price: ${stock.get('current_price', '?')}
            - P/E ratio: {stock.get('pe_ratio', '?')}
            - Dividend yield: {stock.get('dividend_yield', '?')}%
            - 52-week high: ${stock.get('fifty_two_week_high', '?')}
            - 52-week low: ${stock.get('fifty_two_week_low', '?')}
            - Exchange: {stock.get('exchange', '?')}
        """)

    return textwrap.dedent(f"""\
        Write a detailed company analysis HTML document for a financial
        institution's knowledge base.  The document should resemble a
        sell-side equity research summary, 5-7 paragraphs.

        Use this exact HTML structure (no CSS, no scripts):
        <!DOCTYPE html>
        <html>
        <head><title>Company Analysis - {name}</title></head>
        <body>
        <h1>{name} ({ticker}) - Company Analysis Report</h1>
        <p>...</p>
        ...
        </body>
        </html>

        Company facts:
        - Name: {name}
        - Ticker: {ticker}
        - Industry: {industry}
        - Sector: {sector}
        - Market cap: ${mcap}B
        - Headquarters: {hq}
        - Founded: {founded}
        - CEO: {ceo}
        - Employees: {employees}
        - Annual revenue: ${revenue}B
        {stock_info}

        Cover: business overview, competitive position, financial performance,
        growth drivers, risks, and investment outlook.
        Reference the ticker symbol throughout.
        Output only the HTML — no markdown fences or commentary.
    """)


def _sector_analysis_prompt(
    sector: str,
    companies_in_sector: List[Dict[str, str]],
) -> str:
    refs = []
    for c in companies_in_sector[:8]:
        refs.append(f"  - {c.get('name', '?')} ({c.get('ticker_symbol', '?')})")

    return textwrap.dedent(f"""\
        Write a sector/market analysis HTML document for a financial
        institution's knowledge base.  The document should resemble a
        quarterly market outlook, 5-7 paragraphs.

        Use this exact HTML structure (no CSS, no scripts):
        <!DOCTYPE html>
        <html>
        <head><title>Market Analysis - {sector} Sector</title></head>
        <body>
        <h1>{sector} Sector Market Analysis</h1>
        <p>...</p>
        ...
        </body>
        </html>

        Companies to reference by name and ticker:
        {chr(10).join(refs)}

        Cover: sector performance, key trends, notable companies, risks,
        valuation considerations, and forward outlook.
        Output only the HTML — no markdown fences or commentary.
    """)


def _investment_guide_prompt(title: str) -> str:
    return textwrap.dedent(f"""\
        Write an investment strategy / financial planning guide HTML document
        for a financial institution's knowledge base.  The document should be
        educational and actionable, 6-9 paragraphs.

        Use this exact HTML structure (no CSS, no scripts):
        <!DOCTYPE html>
        <html>
        <head><title>{title}</title></head>
        <body>
        <h1>{title}</h1>
        <p>...</p>
        ...
        </body>
        </html>

        Topic: {title}

        Provide practical guidance with specific numbers, percentages, and
        examples where appropriate.
        Output only the HTML — no markdown fences or commentary.
    """)


def _bank_profile_prompt(bank: Dict[str, str]) -> str:
    name = bank.get("name", "Unknown Bank")
    hq = bank.get("headquarters", "")
    btype = bank.get("bank_type", "")
    assets = bank.get("total_assets_billions", "")
    est = bank.get("established_year", "")
    routing = bank.get("routing_number", "")
    swift = bank.get("swift_code", "")

    return textwrap.dedent(f"""\
        Write a bank profile HTML document for a financial institution's
        knowledge base.  The document should resemble an institutional
        overview, 5-6 paragraphs.

        Use this exact HTML structure (no CSS, no scripts):
        <!DOCTYPE html>
        <html>
        <head><title>Bank Profile - {name}</title></head>
        <body>
        <h1>{name} - Institutional Overview</h1>
        <p>...</p>
        ...
        </body>
        </html>

        Bank facts:
        - Name: {name}
        - Headquarters: {hq}
        - Type: {btype}
        - Total assets: ${assets}B
        - Established: {est}
        - Routing number: {routing}
        - SWIFT code: {swift}

        Cover: history, services, financial strength, branch network,
        technology, and community involvement.
        Output only the HTML — no markdown fences or commentary.
    """)


def _regulatory_doc_prompt(title: str) -> str:
    return textwrap.dedent(f"""\
        Write a regulatory/compliance document HTML page for a financial
        institution's knowledge base.  The document should be authoritative
        and detailed, 6-8 paragraphs.

        Use this exact HTML structure (no CSS, no scripts):
        <!DOCTYPE html>
        <html>
        <head><title>{title}</title></head>
        <body>
        <h1>{title}</h1>
        <p>...</p>
        ...
        </body>
        </html>

        Topic: {title}

        Reference specific regulations, acts, and compliance frameworks.
        Provide practical guidance for banking professionals.
        Output only the HTML — no markdown fences or commentary.
    """)


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------


def _call_llm(prompt: str, cfg: GeneratorConfig) -> str:
    """Call the Databricks foundation model endpoint via the OpenAI SDK.

    Uses the ``openai`` client bundled with ``databricks-sdk`` which
    handles workspace authentication automatically on-cluster.
    """
    import os

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=os.environ.get("DATABRICKS_PROFILE"))
    client = w.serving_endpoints.get_open_ai_client()

    response = client.chat.completions.create(
        model=cfg.llm_endpoint,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial content writer producing HTML documents "
                    "for a bank's internal knowledge base.  Output only valid HTML "
                    "with no markdown fences, no code blocks, no commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.7,
    )
    text = response.choices[0].message.content
    # Strip any accidental markdown fences
    text = re.sub(r"^```html?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


# ---------------------------------------------------------------------------
# Dry-run HTML templates (no LLM needed)
# ---------------------------------------------------------------------------


def _dry_run_customer_profile(
    customer: Dict[str, str],
    accounts: List[Dict[str, str]],
    holdings: List[Dict[str, str]],
    stocks: Dict[str, Dict[str, str]],
    companies: Dict[str, Dict[str, str]],
    bank_names: Dict[str, str],
    context: Dict[str, object],
) -> str:
    name = f"{customer['first_name']} {customer['last_name']}"
    cid = customer["customer_id"]
    risk = customer.get("risk_profile") or "not assessed"
    income = customer.get("annual_income", "unknown")
    credit = customer.get("credit_score", "unknown")
    city = customer.get("city", "")
    state = customer.get("state", "")
    first = customer["first_name"]

    holding_mentions = []
    for h in holdings:
        sid = h.get("stock_id", "")
        s = stocks.get(sid, {})
        co = companies.get(s.get("company_id", ""), {})
        ticker = s.get("ticker", "???")
        co_name = co.get("name", "Unknown Company")
        holding_mentions.append(f"{co_name} ({ticker})")

    holdings_text = ", ".join(holding_mentions) if holding_mentions else "no current holdings"
    bank_text = ", ".join(
        bank_names.get(a.get("bank_id", ""), "their bank") for a in accounts
    ) if accounts else "no bank on file"

    sectors_text = ", ".join(context["top_sectors"]) if context["top_sectors"] else "multiple"

    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html>
        <head>
            <title>Customer Profile - {name}</title>
        </head>
        <body>
            <h1>Customer Profile: {name}</h1>

            <p>{name} (Customer ID: {cid}) is a client based in {city}, {state} with a {risk.lower()} risk profile. With an annual income of ${income} and a credit score of {credit}, {first} represents a valued member of our client base at {bank_text}.</p>

            <p>{first} maintains an investment portfolio that includes positions in {holdings_text}. The portfolio shows a concentration in {sectors_text} sectors, reflecting a {context['style_label']} investment approach focused on {context['style_elaboration']}.</p>

            <p>In recent advisory sessions, the advising team has noted that {first} {context['observations'][0]}. Additionally, {first} {context['observations'][1]}. The client's primary financial goal centers on {context['goal']}.</p>

            <p>During portfolio reviews, {first} has expressed particular interest in {context['interests'][0]} and {context['interests'][1]}. These stated preferences present opportunities for targeted investment recommendations aligned with the client's evolving financial objectives.</p>

            <p>As a long-term client, {first} has demonstrated consistent engagement with the bank's wealth management services. Regular reviews with a financial advisor ensure the portfolio remains aligned with stated goals and risk tolerance levels.</p>
        </body>
        </html>
    """)


def _dry_run_company_analysis(
    company: Dict[str, str],
    stock: Optional[Dict[str, str]],
) -> str:
    name = company.get("name", "Unknown")
    ticker = company.get("ticker_symbol", "???")
    sector = company.get("sector", "Unknown")
    mcap = company.get("market_cap_billions", "?")
    hq = company.get("headquarters", "Unknown")
    ceo = company.get("ceo", "Unknown")
    revenue = company.get("annual_revenue_billions", "?")
    employees = company.get("employee_count", "?")
    founded = company.get("founded_year", "?")

    price_text = ""
    if stock:
        price_text = (
            f"{name} currently trades at ${stock.get('current_price', '?')} "
            f"with a P/E ratio of {stock.get('pe_ratio', '?')} and a "
            f"dividend yield of {stock.get('dividend_yield', '?')}%. "
            f"The 52-week range spans ${stock.get('fifty_two_week_low', '?')} "
            f"to ${stock.get('fifty_two_week_high', '?')}."
        )

    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html>
        <head>
            <title>Company Analysis - {name}</title>
        </head>
        <body>
            <h1>{name} ({ticker}) - Company Analysis Report</h1>

            <p>{name}, trading under the ticker symbol {ticker}, is a {sector.lower()} company headquartered in {hq}. Founded in {founded}, the company has grown to employ approximately {employees} professionals and generates annual revenue of ${revenue} billion with a market capitalization of ${mcap} billion.</p>

            <p>Under the leadership of CEO {ceo}, {name} has established a strong competitive position within the {sector.lower()} sector. The company's strategic focus on innovation and operational efficiency has enabled consistent growth and market share expansion.</p>

            <p>{price_text if price_text else f'{name} maintains a solid market valuation reflecting investor confidence in its growth trajectory and business model sustainability.'}</p>

            <p>From an investment perspective, {ticker} offers exposure to the {sector.lower()} sector with a balanced risk-return profile. Analysts point to the company's strong revenue growth, expanding margins, and market leadership as key factors supporting the investment thesis.</p>

            <p>Key risks include competitive pressures within the {sector.lower()} industry, potential regulatory changes, and macroeconomic sensitivity. However, {name}'s diversified business model and strong balance sheet position the company well to navigate various market environments.</p>
        </body>
        </html>
    """)


def _dry_run_sector_analysis(
    sector: str,
    companies_in_sector: List[Dict[str, str]],
) -> str:
    mentions = []
    for c in companies_in_sector[:6]:
        mentions.append(f"{c.get('name', '?')} ({c.get('ticker_symbol', '?')})")
    mentions_text = ", ".join(mentions) if mentions else "various industry participants"

    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html>
        <head>
            <title>Market Analysis - {sector} Sector</title>
        </head>
        <body>
            <h1>{sector} Sector Market Analysis</h1>

            <p>The {sector.lower()} sector continues to present significant investment opportunities driven by evolving market dynamics, technological innovation, and shifting consumer preferences. Notable companies in this space include {mentions_text}.</p>

            <p>Sector performance has been influenced by macroeconomic conditions, regulatory developments, and competitive dynamics. Companies with strong balance sheets and innovative business models have outperformed peers, demonstrating the importance of fundamental analysis in security selection.</p>

            <p>Valuation metrics across the {sector.lower()} sector reflect varied investor expectations for growth and profitability. Forward-looking investors should consider both cyclical factors and structural trends when evaluating opportunities within this space.</p>

            <p>Key trends shaping the {sector.lower()} sector include digital transformation, sustainability initiatives, and evolving regulatory frameworks. Companies positioned to capitalize on these trends are likely to generate superior long-term returns for investors.</p>

            <p>Looking ahead, the {sector.lower()} sector outlook remains cautiously optimistic. While near-term headwinds persist, the long-term growth trajectory is supported by fundamental demand drivers and ongoing innovation across the industry.</p>
        </body>
        </html>
    """)


def _dry_run_investment_guide(title: str) -> str:
    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
        </head>
        <body>
            <h1>{title}</h1>

            <p>Effective investment strategy requires a disciplined approach that balances risk and return objectives while accounting for individual circumstances, time horizons, and financial goals. This guide provides a framework for making informed investment decisions in the context of {title.lower()}.</p>

            <p>Asset allocation represents the most critical decision in portfolio construction, accounting for approximately 90% of portfolio return variability according to academic research. A well-diversified portfolio should span multiple asset classes including equities, fixed income, real estate, and alternative investments.</p>

            <p>Risk management is an integral component of any sound investment strategy. Understanding your risk tolerance, maintaining appropriate diversification, and implementing systematic rebalancing disciplines help ensure portfolios remain aligned with stated objectives through various market conditions.</p>

            <p>Tax efficiency can significantly impact net investment returns over time. Strategies such as tax-loss harvesting, asset location optimization, and Roth conversion planning can enhance after-tax portfolio performance without requiring additional market risk.</p>

            <p>Regular portfolio review and adjustment ensures investment strategies remain appropriate as personal circumstances evolve and market conditions change. Annual reviews with a qualified financial advisor are recommended to assess progress toward goals and make necessary adjustments.</p>

            <p>Cost management represents another crucial element of investment success. Favoring low-cost index funds and ETFs for core portfolio positions while maintaining an overall expense ratio below 0.50% helps ensure that fees do not significantly erode long-term returns.</p>
        </body>
        </html>
    """)


def _dry_run_bank_profile(bank: Dict[str, str]) -> str:
    name = bank.get("name", "Unknown Bank")
    hq = bank.get("headquarters", "Unknown")
    btype = bank.get("bank_type", "Commercial")
    assets = bank.get("total_assets_billions", "?")
    est = bank.get("established_year", "?")
    routing = bank.get("routing_number", "?")
    swift = bank.get("swift_code", "?")

    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bank Profile - {name}</title>
        </head>
        <body>
            <h1>{name} - Institutional Overview</h1>

            <p>{name}, established in {est}, is a {btype.lower()} banking institution headquartered in {hq}. With total assets exceeding ${assets} billion, the bank serves a diverse client base of individual consumers, small businesses, and corporate clients.</p>

            <p>The institution operates under routing number {routing} and SWIFT code {swift}, facilitating both domestic and international transactions. {name} specializes in {btype.lower()} banking services including deposit accounts, lending products, wealth management, and treasury services.</p>

            <p>{name}'s investment philosophy emphasizes prudent risk management and client-centric service delivery. The bank's wealth management division provides portfolio management, estate planning, and retirement planning services tailored to individual client needs and objectives.</p>

            <p>The bank maintains strong capital ratios well above regulatory requirements, demonstrating financial soundness and the ability to weather economic uncertainties. A well-diversified loan portfolio and conservative underwriting standards contribute to consistently low non-performing loan ratios.</p>

            <p>{name} is recognized for its commitment to community development and financial inclusion. The institution actively invests in community development projects, affordable housing initiatives, and financial literacy programs across its service area.</p>
        </body>
        </html>
    """)


def _dry_run_regulatory_doc(title: str) -> str:
    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
        </head>
        <body>
            <h1>{title}</h1>

            <p>The banking industry faces an increasingly complex regulatory environment, with federal and state regulators implementing new requirements while maintaining vigilant oversight of existing regulations. Financial institutions must navigate this landscape carefully to ensure compliance while maintaining operational efficiency.</p>

            <p>The Bank Secrecy Act (BSA) and Anti-Money Laundering (AML) requirements continue to represent major compliance priorities. Banks must maintain robust Know Your Customer (KYC) programs that verify customer identities, understand the nature and purpose of customer relationships, and conduct ongoing monitoring.</p>

            <p>Capital requirements under Basel III continue to shape bank operations and strategic decision-making. The Common Equity Tier 1 (CET1) ratio requirement ensures that banks maintain sufficient loss-absorbing capacity to withstand economic stress scenarios.</p>

            <p>Consumer protection regulations require careful attention to product disclosures, fair lending practices, and complaint resolution procedures. Violations can result in substantial penalties, reputational damage, and mandatory consumer restitution.</p>

            <p>Cybersecurity and data privacy represent increasingly critical compliance areas. Financial institutions must implement multi-layered security controls, conduct regular security assessments, and maintain incident response plans in accordance with applicable regulations.</p>

            <p>Compliance costs continue to rise, with industry estimates suggesting that banks spend 3-5% of operating expenses on compliance activities. However, effective compliance programs protect banks from far more costly enforcement actions and reputational damage.</p>
        </body>
        </html>
    """)


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _run_llm_tasks(
    tasks: List[Tuple[GeneratedDocument, Optional[str]]],
    cfg: GeneratorConfig,
) -> None:
    """Execute LLM calls for all pending tasks in parallel.

    Each task is a ``(document, prompt)`` pair.  When *prompt* is not
    ``None`` the document's ``html_content`` is filled by calling the
    LLM endpoint.  Dry-run tasks (prompt is ``None``) are skipped — their
    HTML is already set.
    """
    llm_items = [(i, prompt) for i, (_, prompt) in enumerate(tasks) if prompt]
    if not llm_items:
        return

    total = len(tasks)
    lock = threading.Lock()
    done = [0]

    def _run(idx: int, prompt: str) -> Tuple[int, str]:
        return idx, _call_llm(prompt, cfg)

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {pool.submit(_run, i, p): i for i, p in llm_items}
        for future in as_completed(futures):
            idx, html = future.result()
            tasks[idx][0].html_content = html
            with lock:
                done[0] += 1
                logger.info(
                    "  [%d/%d] %s", done[0], total, tasks[idx][0].filename,
                )


def generate_documents(cfg: GeneratorConfig) -> List[GeneratedDocument]:
    """Generate all HTML documents.

    In dry-run mode, documents are built sequentially using templates.
    In live mode, entity selection and prompt construction happen
    sequentially (deterministic via *seed*), then all LLM calls are
    dispatched to a thread pool controlled by ``cfg.max_workers``.

    Returns a list of ``GeneratedDocument`` objects ready for chunking
    and embedding.
    """
    rng = random.Random(cfg.seed)
    entities = load_entities(cfg)

    # Build lookup tables
    stocks_by_id = {s["stock_id"]: s for s in entities["stocks"]}
    stocks_by_company = {s["company_id"]: s for s in entities["stocks"]}
    companies_by_id = {c["company_id"]: c for c in entities["companies"]}
    bank_names = {b["bank_id"]: b["name"] for b in entities["banks"]}
    accounts_by_customer: Dict[str, List[Dict[str, str]]] = {}
    for a in entities["accounts"]:
        accounts_by_customer.setdefault(a["customer_id"], []).append(a)
    holdings_by_account: Dict[str, List[Dict[str, str]]] = {}
    for h in entities["portfolio_holdings"]:
        holdings_by_account.setdefault(h["account_id"], []).append(h)

    total = cfg.total_documents
    mode = "dry-run" if cfg.dry_run else f"LLM, {cfg.max_workers} workers"

    # Collect (document, prompt_or_None) pairs.  In dry-run mode the
    # HTML is filled immediately; in live mode it stays empty until the
    # parallel phase fills it via the LLM.
    tasks: List[Tuple[GeneratedDocument, Optional[str]]] = []

    # ----- Customer profiles -----
    logger.info("Generating %d customer profiles (%s)...", cfg.num_customer_profiles, mode)
    selected_customers = _select_customers(entities, cfg.num_customer_profiles, rng)
    for cust in selected_customers:
        cid = cust["customer_id"]
        name = f"{cust['first_name']} {cust['last_name']}"
        accts = accounts_by_customer.get(cid, [])
        all_holdings: List[Dict[str, str]] = []
        for a in accts:
            all_holdings.extend(holdings_by_account.get(a["account_id"], []))

        context = _build_customer_context(
            rng, cust, all_holdings, stocks_by_id, companies_by_id,
        )

        if cfg.dry_run:
            html = _dry_run_customer_profile(
                cust, accts, all_holdings, stocks_by_id, companies_by_id,
                bank_names, context,
            )
            prompt = None
        else:
            html = ""
            prompt = _customer_profile_prompt(
                cust, accts, all_holdings, stocks_by_id, companies_by_id,
                bank_names, context,
            )

        # Build entity references
        ref_companies = []
        ref_tickers = []
        for h in all_holdings:
            s = stocks_by_id.get(h.get("stock_id", ""), {})
            co = companies_by_id.get(s.get("company_id", ""), {})
            if co.get("name"):
                ref_companies.append(co["name"])
            if s.get("ticker"):
                ref_tickers.append(s["ticker"])
        for a in accts:
            bname = bank_names.get(a.get("bank_id", ""))
            if bname:
                ref_companies.append(bname)

        filename = f"customer_profile_{_slugify(name)}.html"
        doc = GeneratedDocument(
            filename=filename,
            document_type="customer_profile",
            title=f"Customer Profile - {name}",
            html_content=html,
            entity_references=EntityReferences(
                customers=[name],
                companies=ref_companies,
                stock_tickers=ref_tickers,
            ),
        )
        tasks.append((doc, prompt))
        if cfg.dry_run:
            logger.info(
                "  [%d/%d] %s (style=%s)", len(tasks), total, filename, context["style"],
            )

    # ----- Company analyses -----
    logger.info("Generating %d company analyses (%s)...", cfg.num_company_analyses, mode)
    selected_companies = _select_companies(entities, cfg.num_company_analyses, rng)
    for comp in selected_companies:
        name = comp.get("name", "Unknown")
        ticker = comp.get("ticker_symbol", "???")
        stock = stocks_by_company.get(comp["company_id"])

        if cfg.dry_run:
            html = _dry_run_company_analysis(comp, stock)
            prompt = None
        else:
            html = ""
            prompt = _company_analysis_prompt(comp, stock)

        filename = f"company_analysis_{_slugify(name)}.html"
        doc = GeneratedDocument(
            filename=filename,
            document_type="company_analysis",
            title=f"Company Analysis - {name}",
            html_content=html,
            entity_references=EntityReferences(
                companies=[name],
                stock_tickers=[ticker],
            ),
        )
        tasks.append((doc, prompt))
        if cfg.dry_run:
            logger.info("  [%d/%d] %s", len(tasks), total, filename)

    # ----- Sector / market analyses -----
    companies_by_sector: Dict[str, List[Dict[str, str]]] = {}
    for c in entities["companies"]:
        companies_by_sector.setdefault(c.get("sector", "Unknown"), []).append(c)

    available_sectors = list(companies_by_sector.keys())
    rng.shuffle(available_sectors)
    sectors_to_cover = available_sectors[: cfg.num_sector_analyses]

    logger.info("Generating %d sector analyses (%s)...", cfg.num_sector_analyses, mode)
    for sector in sectors_to_cover:
        cos = companies_by_sector.get(sector, [])
        if cfg.dry_run:
            html = _dry_run_sector_analysis(sector, cos)
            prompt = None
        else:
            html = ""
            prompt = _sector_analysis_prompt(sector, cos)

        ref_companies = [c.get("name", "") for c in cos[:8]]
        ref_tickers = [c.get("ticker_symbol", "") for c in cos[:8]]

        filename = f"market_analysis_{_slugify(sector)}_sector.html"
        doc = GeneratedDocument(
            filename=filename,
            document_type="market_analysis",
            title=f"Market Analysis - {sector} Sector",
            html_content=html,
            entity_references=EntityReferences(
                companies=ref_companies,
                stock_tickers=ref_tickers,
            ),
        )
        tasks.append((doc, prompt))
        if cfg.dry_run:
            logger.info("  [%d/%d] %s", len(tasks), total, filename)

    # ----- Investment guides -----
    logger.info("Generating %d investment guides (%s)...", cfg.num_investment_guides, mode)
    guides = list(_INVESTMENT_GUIDE_TOPICS)
    rng.shuffle(guides)
    for title, doc_type in guides[: cfg.num_investment_guides]:
        if cfg.dry_run:
            html = _dry_run_investment_guide(title)
            prompt = None
        else:
            html = ""
            prompt = _investment_guide_prompt(title)

        filename = f"{_slugify(title)}.html"
        doc = GeneratedDocument(
            filename=filename,
            document_type=doc_type,
            title=title,
            html_content=html,
            entity_references=EntityReferences(),
        )
        tasks.append((doc, prompt))
        if cfg.dry_run:
            logger.info("  [%d/%d] %s", len(tasks), total, filename)

    # ----- Bank profiles -----
    logger.info("Generating %d bank profiles (%s)...", cfg.num_bank_profiles, mode)
    selected_banks = _select_banks(entities, cfg.num_bank_profiles, rng)
    for bank in selected_banks:
        name = bank.get("name", "Unknown Bank")

        if cfg.dry_run:
            html = _dry_run_bank_profile(bank)
            prompt = None
        else:
            html = ""
            prompt = _bank_profile_prompt(bank)

        filename = f"bank_profile_{_slugify(name)}.html"
        doc = GeneratedDocument(
            filename=filename,
            document_type="bank_profile",
            title=f"Bank Profile - {name}",
            html_content=html,
            entity_references=EntityReferences(
                companies=[name],
            ),
        )
        tasks.append((doc, prompt))
        if cfg.dry_run:
            logger.info("  [%d/%d] %s", len(tasks), total, filename)

    # ----- Regulatory / compliance docs -----
    logger.info("Generating %d regulatory docs (%s)...", cfg.num_regulatory_docs, mode)
    reg_topics = list(_REGULATORY_TOPICS)
    rng.shuffle(reg_topics)
    for title, doc_type in reg_topics[: cfg.num_regulatory_docs]:
        if cfg.dry_run:
            html = _dry_run_regulatory_doc(title)
            prompt = None
        else:
            html = ""
            prompt = _regulatory_doc_prompt(title)

        filename = f"{_slugify(title)}.html"
        doc = GeneratedDocument(
            filename=filename,
            document_type=doc_type,
            title=title,
            html_content=html,
            entity_references=EntityReferences(),
        )
        tasks.append((doc, prompt))
        if cfg.dry_run:
            logger.info("  [%d/%d] %s", len(tasks), total, filename)

    # Phase 2: execute LLM calls in parallel (live mode only)
    if not cfg.dry_run:
        logger.info("Submitting %d LLM calls (%d workers)...", len(tasks), cfg.max_workers)
        _run_llm_tasks(tasks, cfg)

    documents = [doc for doc, _ in tasks]
    logger.info("Generated %d HTML documents", len(documents))
    return documents
