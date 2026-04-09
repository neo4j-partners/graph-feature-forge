"""Generator functions for each entity type.

Each generator takes a config and a seeded random.Random instance, and
returns a list of Pydantic model instances. Generators must be called in
dependency order: banks -> companies -> stocks -> customers -> accounts ->
portfolio holdings -> transactions.
"""

from __future__ import annotations

import math
import random
import string
from datetime import date, timedelta

import numpy as np

from faker import Faker

from .config import GeneratorConfig
from .models import (
    Account,
    Bank,
    Company,
    Customer,
    PortfolioHolding,
    Stock,
    Transaction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

US_STATES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
]

BANK_NAME_PREFIXES = [
    "First",
    "National",
    "Pacific",
    "Mountain",
    "Atlantic",
    "Southern",
    "Midwest",
    "Great Lakes",
    "Capital",
    "Northwest",
    "Sunshine",
    "Rocky Mountain",
    "Empire",
    "Liberty",
    "Coastal",
    "Metropolitan",
    "Heritage",
    "Pioneer",
    "Evergreen",
    "Cornerstone",
    "Summit",
    "Valley",
    "Riverbend",
    "Horizon",
    "Gateway",
    "Silverado",
    "Compass",
    "Anchor",
    "Bridgeview",
    "Eastside",
    "Westwood",
    "Northern",
    "Delta",
    "Pinnacle",
    "Keystone",
    "Blue Ridge",
    "Prairie",
    "Sunbelt",
    "Lakeside",
    "Mountain View",
    "Colonial",
    "Frontier",
    "Redwood",
    "Harbourside",
    "Continental",
    "Timberline",
    "Magnolia",
    "High Desert",
    "Crossroads",
    "Lighthouse",
]

BANK_NAME_SUFFIXES = [
    "Trust",
    "Bank",
    "Financial",
    "Banking Corp",
    "Federal Bank",
    "Credit Union",
    "Savings & Loan",
    "Trust Company",
    "Banking Group",
    "Commerce Bank",
    "National Bank",
    "Bank & Trust",
]

BANK_TYPES = ["Commercial", "Regional", "Credit Union", "Savings", "Community"]
BANK_TYPE_WEIGHTS = [25, 30, 10, 10, 25]

INDUSTRIES_BY_SECTOR: dict[str, list[str]] = {
    "Technology": [
        "Software",
        "Cloud Services",
        "Semiconductors",
        "Cybersecurity",
        "AI Services",
        "Data Infrastructure",
        "IoT Devices",
        "EdTech",
        "Banking Software",
        "Computing",
    ],
    "Healthcare": [
        "Biotechnology",
        "Pharmaceuticals",
        "Medical Equipment",
        "Healthcare Providers",
        "Digital Health",
        "Health Tech",
        "Biopharmaceuticals",
        "Healthcare Technology",
        "Genetic Testing",
        "Personalized Medicine",
        "Neurotechnology",
    ],
    "Financial Services": [
        "Banking",
        "Asset Management",
        "Insurance",
        "Financial Technology",
        "Retail Investment",
        "Cryptocurrency",
    ],
    "Energy": [
        "Oil & Gas",
        "Renewable Energy",
        "Solar",
        "Nuclear Energy",
        "Alternative Energy",
        "Energy Storage",
        "Marine Energy",
    ],
    "Consumer Discretionary": [
        "Retail",
        "Automotive",
        "Hotels",
        "Restaurants",
        "Consumer Electronics",
        "Luxury Goods",
        "Home Furnishings",
        "Fitness",
        "Gaming",
        "Textiles",
        "Wearable Tech",
    ],
    "Consumer Staples": [
        "Consumer Products",
        "Beverages",
        "Supermarkets",
        "Coffee",
        "Agriculture",
        "Food Science",
        "Alternative Proteins",
    ],
    "Industrials": [
        "Aerospace",
        "Transportation",
        "Construction",
        "Defense",
        "Shipping",
        "Manufacturing Tech",
        "Robotics",
        "Unmanned Systems",
        "Environmental Tech",
        "Space Industry",
    ],
    "Communication Services": [
        "Telecommunications",
        "Media",
        "Film & TV",
        "Entertainment",
        "Social Media",
        "Internet",
        "Communications",
    ],
    "Materials": [
        "Mining",
        "Chemicals",
        "Steel",
        "Paper & Packaging",
        "Advanced Materials",
        "Materials Science",
    ],
    "Real Estate": [
        "Real Estate",
        "Sustainable Construction",
    ],
    "Utilities": [
        "Utilities",
        "Energy Tech",
        "Environmental",
    ],
}

# P/E ratio ranges by sector (low, high)
PE_RANGES: dict[str, tuple[float, float]] = {
    "Technology": (22.0, 55.0),
    "Healthcare": (18.0, 50.0),
    "Financial Services": (10.0, 25.0),
    "Energy": (8.0, 20.0),
    "Consumer Discretionary": (14.0, 30.0),
    "Consumer Staples": (14.0, 22.0),
    "Industrials": (12.0, 28.0),
    "Communication Services": (16.0, 30.0),
    "Materials": (10.0, 20.0),
    "Real Estate": (14.0, 30.0),
    "Utilities": (12.0, 20.0),
}

TRANSACTION_TYPES = ["Transfer", "Payment", "Deposit", "Withdrawal"]
TRANSACTION_TYPE_WEIGHTS = [35, 30, 20, 15]

TRANSACTION_DESCRIPTIONS = [
    "Payment for services",
    "Investment transfer",
    "Rent payment",
    "Utility bill",
    "Contract payment",
    "Grocery shopping",
    "Car payment",
    "Insurance premium",
    "Tuition payment",
    "Medical expenses",
    "Business invoice",
    "Restaurant bill",
    "Phone bill",
    "Consulting fee",
    "Internet service",
    "Property tax",
    "Gas bill",
    "Legal fees",
    "Home repair",
    "Quarterly tax",
    "Gym membership",
    "Equipment purchase",
    "Software subscription",
    "Loan repayment",
    "Credit card payment",
    "Contract work",
    "Delivery service",
    "Equipment lease",
    "Maintenance fee",
    "Marketing services",
    "Event ticket",
    "Security deposit",
    "Cleaning service",
    "Accounting fees",
    "Design work",
    "Printing services",
    "Training program",
    "Travel expenses",
    "Catering service",
    "Refund",
    "Subscription renewal",
    "Course enrollment",
    "Auto repair",
    "Appliance purchase",
    "Moving expenses",
    "Furniture",
    "Landscaping",
    "Pool maintenance",
    "HVAC repair",
    "Plumbing work",
    "Painting services",
    "Roofing repair",
    "Window cleaning",
    "Office supplies",
    "Network upgrade",
    "Phone system",
    "Conference room",
    "App development",
    "Database setup",
    "Cloud migration",
    "Data backup",
    "Website design",
    "SEO services",
]

EMPLOYMENT_STATUSES = ["Employed", "Self-Employed", "Retired", "Unemployed"]
EMPLOYMENT_WEIGHTS = [70, 15, 10, 5]

RISK_PROFILES = ["Aggressive", "Conservative", "Moderate"]

# Sector preference weights per risk profile. Higher weight = more likely to
# hold stocks in that sector. Non-zero floors ensure realistic noise.
SECTOR_PREFERENCE_WEIGHTS: dict[str, dict[str, float]] = {
    "Aggressive": {
        "Technology": 5.0,
        "Healthcare": 4.0,
        "Communication Services": 3.5,
        "Consumer Discretionary": 2.5,
        "Energy": 1.5,
        "Financial Services": 1.5,
        "Industrials": 1.0,
        "Materials": 1.0,
        "Consumer Staples": 0.5,
        "Real Estate": 0.5,
        "Utilities": 0.3,
    },
    "Conservative": {
        "Utilities": 5.0,
        "Consumer Staples": 4.5,
        "Real Estate": 4.0,
        "Materials": 2.5,
        "Energy": 2.5,
        "Financial Services": 2.0,
        "Industrials": 1.5,
        "Consumer Discretionary": 1.0,
        "Healthcare": 0.8,
        "Communication Services": 0.5,
        "Technology": 0.3,
    },
    "Moderate": {
        "Financial Services": 4.0,
        "Industrials": 3.5,
        "Consumer Discretionary": 3.0,
        "Healthcare": 2.5,
        "Technology": 2.0,
        "Energy": 2.0,
        "Consumer Staples": 2.0,
        "Materials": 2.0,
        "Communication Services": 1.5,
        "Real Estate": 1.5,
        "Utilities": 1.5,
    },
}


def _create_faker(seed: int) -> Faker:
    """Create a seeded Faker instance for reproducible generation."""
    fake = Faker()
    fake.seed_instance(seed)
    return fake


def _log_normal_sample(
    rng: random.Random,
    median: float,
    sigma: float,
    lo: float,
    hi: float,
) -> float:
    """Draw from a log-normal distribution, clamped to [lo, hi]."""
    mu = math.log(median)
    value = rng.lognormvariate(mu, sigma)
    return max(lo, min(hi, value))


def _normal_int_sample(
    rng: random.Random,
    mean: float,
    std: float,
    lo: int,
    hi: int,
) -> int:
    """Draw from a normal distribution, clamped and rounded to int."""
    value = rng.gauss(mean, std)
    return max(lo, min(hi, round(value)))


def _random_date(
    rng: random.Random,
    start: date,
    end: date,
) -> date:
    """Return a random date between start and end inclusive."""
    delta_days = (end - start).days
    return start + timedelta(days=rng.randint(0, delta_days))


def _generate_routing_number(rng: random.Random) -> str:
    """Generate a plausible 9-digit routing number."""
    return "".join(str(rng.randint(0, 9)) for _ in range(9))


def _generate_swift_code(rng: random.Random) -> str:
    """Generate a plausible 8-character SWIFT code."""
    letters = "".join(rng.choices(string.ascii_uppercase, k=4))
    country = "US"
    location = "".join(rng.choices(string.ascii_uppercase + string.digits, k=2))
    return f"{letters}{country}{location}"


def _select_stocks_by_risk(
    rng_np: np.random.Generator,
    stocks: list[Stock],
    sector_by_company: dict[str, str],
    num_stocks: int,
    risk_profile: str,
    strength: float = 1.0,
) -> list[Stock]:
    """Select stocks with sector bias based on risk profile.

    *strength* controls correlation: 0.0 = uniform random, 1.0 = default
    bias, >1.0 = stronger bias.  Weights are raised to the power of
    *strength* so that strength=0 collapses all weights to 1.0.

    Uses numpy for weighted sampling without replacement.
    """
    prefs = SECTOR_PREFERENCE_WEIGHTS.get(risk_profile, {})

    raw = np.array(
        [
            prefs.get(sector_by_company.get(s.company_id, ""), 1.0) ** strength
            for s in stocks
        ]
    )
    probs = raw / raw.sum()

    n = min(num_stocks, len(stocks))
    indices = rng_np.choice(len(stocks), size=n, replace=False, p=probs)
    return [stocks[i] for i in indices]


# ---------------------------------------------------------------------------
# Entity generators
# ---------------------------------------------------------------------------


def generate_banks(config: GeneratorConfig, rng: random.Random) -> list[Bank]:
    """Generate bank entities. No dependencies."""
    fake = _create_faker(config.random_seed)
    banks: list[Bank] = []

    used_names: set[str] = set()
    for i in range(config.num_banks):
        bank_id = f"B{i + 1:03d}"

        # Generate a unique bank name
        while True:
            prefix = rng.choice(BANK_NAME_PREFIXES)
            suffix = rng.choice(BANK_NAME_SUFFIXES)
            name = f"{prefix} {suffix}"
            if name not in used_names:
                used_names.add(name)
                break

        city = fake.city()
        state = rng.choice(US_STATES)
        headquarters = f"{city} {state}"

        bank_type = rng.choices(BANK_TYPES, weights=BANK_TYPE_WEIGHTS)[0]

        # Assets: log-uniform spread across 1B-500B
        assets = round(
            _log_normal_sample(
                rng,
                50.0,
                1.0,
                config.bank_assets_min_billions,
                config.bank_assets_max_billions,
            ),
            1,
        )

        established_year = rng.randint(1890, 2005)
        routing_number = _generate_routing_number(rng)
        swift_code = _generate_swift_code(rng)

        banks.append(
            Bank(
                bank_id=bank_id,
                name=name,
                headquarters=headquarters,
                bank_type=bank_type,
                total_assets_billions=assets,
                established_year=established_year,
                routing_number=routing_number,
                swift_code=swift_code,
            )
        )

    return banks


def generate_companies(config: GeneratorConfig, rng: random.Random) -> list[Company]:
    """Generate company entities. No dependencies."""
    fake = _create_faker(config.random_seed + 1)
    companies: list[Company] = []

    sectors = list(INDUSTRIES_BY_SECTOR.keys())
    used_tickers: set[str] = set()

    for i in range(config.num_companies):
        company_id = f"CO{i + 1:03d}"

        sector = rng.choice(sectors)
        industry = rng.choice(INDUSTRIES_BY_SECTOR[sector])

        # Company name: adjective + noun + suffix
        name = fake.company()

        # Ticker: 3-4 uppercase letters, unique
        while True:
            ticker_len = rng.choice([3, 4])
            ticker = "".join(rng.choices(string.ascii_uppercase, k=ticker_len))
            if ticker not in used_tickers:
                used_tickers.add(ticker)
                break

        city = fake.city()
        state = rng.choice(US_STATES)
        headquarters = f"{city} {state}"

        founded_year = rng.randint(1945, 2020)
        ceo = f"{fake.first_name()} {fake.last_name()}"

        # Market cap: log-normal, range 5B-1500B
        market_cap = round(
            _log_normal_sample(rng, 80.0, 0.9, 5.0, 1500.0),
            1,
        )

        # Revenue should be plausible relative to market cap
        # revenue/market_cap ratio: roughly 0.1-0.4 for most companies
        rev_ratio = rng.uniform(0.08, 0.40)
        revenue = round(market_cap * rev_ratio, 1)

        # Employee count correlates loosely with revenue
        employees_per_billion_rev = rng.uniform(800, 5000)
        employee_count = max(500, round(revenue * employees_per_billion_rev))

        companies.append(
            Company(
                company_id=company_id,
                name=name,
                ticker_symbol=ticker,
                industry=industry,
                sector=sector,
                market_cap_billions=market_cap,
                headquarters=headquarters,
                founded_year=founded_year,
                ceo=ceo,
                employee_count=employee_count,
                annual_revenue_billions=revenue,
            )
        )

    return companies


def generate_stocks(
    config: GeneratorConfig,
    rng: random.Random,
    companies: list[Company],
) -> list[Stock]:
    """Generate one stock per company. Depends on companies."""
    stocks: list[Stock] = []

    for i, company in enumerate(companies):
        stock_id = f"S{i + 1:03d}"

        # Stock price relative to market cap and shares outstanding
        # Assume shares outstanding = market_cap_billions * 1B / price
        # Pick a plausible price range based on market cap
        if company.market_cap_billions > 500:
            base_price = rng.uniform(300, 1300)
        elif company.market_cap_billions > 100:
            base_price = rng.uniform(80, 400)
        elif company.market_cap_billions > 30:
            base_price = rng.uniform(30, 150)
        else:
            base_price = rng.uniform(10, 80)

        current_price = round(base_price, 2)

        # Day variations
        day_change_pct = rng.gauss(0, 0.015)
        previous_close = round(current_price / (1 + day_change_pct), 2)
        opening_price = round(
            previous_close + (current_price - previous_close) * rng.uniform(0.2, 0.8),
            2,
        )
        day_high = round(
            max(current_price, opening_price, previous_close)
            * (1 + abs(rng.gauss(0, 0.008))),
            2,
        )
        day_low = round(
            min(current_price, opening_price, previous_close)
            * (1 - abs(rng.gauss(0, 0.008))),
            2,
        )

        # Volume correlates with market cap
        base_volume = int(company.market_cap_billions * rng.uniform(10000, 80000))
        volume = max(100000, base_volume)

        # P/E ratio by sector
        pe_lo, pe_hi = PE_RANGES.get(company.sector, (12.0, 25.0))
        pe_ratio = round(rng.uniform(pe_lo, pe_hi), 1)

        # Dividend yield: tech/growth = 0, value = higher
        if company.sector in ("Technology", "Healthcare"):
            dividend_yield = round(
                rng.choices([0.0, rng.uniform(0.3, 1.5)], weights=[60, 40])[0], 1
            )
        elif company.sector in ("Utilities", "Materials", "Energy"):
            dividend_yield = round(rng.uniform(1.5, 4.5), 1)
        else:
            dividend_yield = round(rng.uniform(0.0, 3.0), 1)

        # 52-week range
        fifty_two_week_high = round(current_price * rng.uniform(1.10, 1.35), 2)
        fifty_two_week_low = round(current_price * rng.uniform(0.65, 0.90), 2)

        exchange = rng.choice(["NYSE", "NASDAQ"])

        stocks.append(
            Stock(
                stock_id=stock_id,
                ticker=company.ticker_symbol,
                company_id=company.company_id,
                current_price=current_price,
                previous_close=previous_close,
                opening_price=opening_price,
                day_high=day_high,
                day_low=day_low,
                volume=volume,
                market_cap_billions=company.market_cap_billions,
                pe_ratio=pe_ratio,
                dividend_yield=dividend_yield,
                fifty_two_week_high=fifty_two_week_high,
                fifty_two_week_low=fifty_two_week_low,
                exchange=exchange,
            )
        )

    return stocks


def generate_customers(
    config: GeneratorConfig,
    rng: random.Random,
) -> tuple[list[Customer], dict[str, str]]:
    """Generate customer entities. No dependencies.

    Returns ``(customers, risk_profile_map)`` where *risk_profile_map*
    maps every customer_id to their true risk profile.  Only
    ``labels_per_class`` customers per class have the profile visible
    in the CSV; the rest get ``risk_profile=""``.
    """
    fake = _create_faker(config.random_seed + 2)
    customers: list[Customer] = []

    # Assign a true risk profile to EVERY customer.  The profile drives
    # income/credit nudges and portfolio sector selection for all
    # customers, but only the labeled subset exposes it in the CSV.
    customer_ids = [f"C{i + 1:04d}" for i in range(config.num_customers)]
    risk_profile_map = {cid: rng.choice(RISK_PROFILES) for cid in customer_ids}

    # Stratified selection of labeled customers (labels_per_class per class)
    labeled_ids: set[str] = set()
    for profile in RISK_PROFILES:
        candidates = [cid for cid, p in risk_profile_map.items() if p == profile]
        sample_size = min(config.labels_per_class, len(candidates))
        labeled_ids.update(rng.sample(candidates, sample_size))

    for i, customer_id in enumerate(customer_ids):
        first_name = fake.first_name()
        last_name = fake.last_name()
        email = f"{first_name.lower()}.{last_name.lower()}@email.com"
        phone = f"555-{rng.randint(100, 999):03d}-{rng.randint(1000, 9999):04d}"

        address = fake.street_address()
        city = fake.city()
        state = rng.choice(US_STATES)
        zip_code = f"{rng.randint(10000, 99999)}"

        # Registration date: spread over 3 years (2018-2021)
        registration_date = _random_date(
            rng,
            date(2018, 1, 1),
            date(2021, 12, 31),
        )

        # Date of birth: ages 25-65 (born 1960-2000 relative to ~2025)
        date_of_birth = _random_date(
            rng,
            date(1960, 1, 1),
            date(2000, 12, 31),
        )

        # Employment status
        employment_status = rng.choices(
            EMPLOYMENT_STATUSES,
            weights=EMPLOYMENT_WEIGHTS,
        )[0]

        # Annual income: log-normal
        annual_income = round(
            _log_normal_sample(
                rng,
                config.income_median,
                config.income_sigma,
                config.income_min,
                config.income_max,
            ),
        )

        # Credit score: normal
        credit_score = _normal_int_sample(
            rng,
            config.credit_score_mean,
            config.credit_score_std,
            config.credit_score_min,
            config.credit_score_max,
        )

        # Apply income/credit nudges based on true risk profile for ALL
        # customers (not just labeled ones).  This ensures unlabeled
        # customers also have realistic attribute correlations.
        true_risk = risk_profile_map[customer_id]
        if true_risk == "Aggressive":
            annual_income = min(
                config.income_max, round(annual_income * rng.uniform(1.05, 1.25))
            )
            credit_score = max(
                config.credit_score_min, credit_score - rng.randint(0, 30)
            )
        elif true_risk == "Conservative":
            credit_score = min(
                config.credit_score_max, credit_score + rng.randint(0, 40)
            )
            annual_income = max(
                config.income_min, round(annual_income * rng.uniform(0.85, 1.05))
            )
        elif true_risk == "Moderate":
            credit_score = _normal_int_sample(
                rng,
                720,
                60,
                config.credit_score_min,
                config.credit_score_max,
            )

        # Retired people tend to have lower income
        if employment_status == "Retired":
            annual_income = max(
                config.income_min, round(annual_income * rng.uniform(0.5, 0.75))
            )
        elif employment_status == "Unemployed":
            annual_income = max(
                config.income_min, round(annual_income * rng.uniform(0.3, 0.6))
            )

        # Only labeled customers expose their profile in the CSV
        risk_profile = true_risk if customer_id in labeled_ids else ""

        customers.append(
            Customer(
                customer_id=customer_id,
                first_name=first_name,
                last_name=last_name,
                email=email,
                phone=phone,
                address=address,
                city=city,
                state=state,
                zip_code=zip_code,
                registration_date=registration_date.isoformat(),
                date_of_birth=date_of_birth.isoformat(),
                risk_profile=risk_profile,
                employment_status=employment_status,
                annual_income=annual_income,
                credit_score=credit_score,
            )
        )

    return customers, risk_profile_map


def generate_accounts(
    config: GeneratorConfig,
    rng: random.Random,
    customers: list[Customer],
    banks: list[Bank],
) -> list[Account]:
    """Generate accounts for each customer. Depends on customers and banks."""
    accounts: list[Account] = []
    account_counter = 0
    account_number_counter = 1001000000

    account_count_weights = [
        config.accounts_weight_one,
        config.accounts_weight_two,
        config.accounts_weight_three,
    ]

    for customer in customers:
        num_accounts = rng.choices([1, 2, 3], weights=account_count_weights)[0]

        # Pick a primary bank for this customer (customers share banks)
        bank = rng.choice(banks)

        # Decide account types. If 1: random. If 2+: ensure variety.
        if num_accounts == 1:
            account_types = [rng.choice(["Checking", "Savings", "Investment"])]
        elif num_accounts == 2:
            account_types = rng.sample(["Checking", "Savings", "Investment"], 2)
        else:
            account_types = ["Checking", "Savings", "Investment"]
            rng.shuffle(account_types)

        for acct_type in account_types:
            account_counter += 1
            account_id = f"A{account_counter:05d}"
            account_number_counter += 1
            account_number = str(account_number_counter)

            # Balance correlates with income
            income_factor = customer.annual_income / config.income_median
            if acct_type == "Checking":
                base_balance = rng.uniform(2000, 20000) * income_factor
                interest_rate = 0.05
            elif acct_type == "Savings":
                base_balance = rng.uniform(5000, 80000) * income_factor
                interest_rate = round(rng.uniform(1.0, 4.0), 2)
            else:  # Investment
                base_balance = rng.uniform(50000, 400000) * income_factor
                interest_rate = 0.00

            balance = round(base_balance, 2)

            # Opened date: around or after registration
            reg_date = date.fromisoformat(customer.registration_date)
            opened_date = _random_date(
                rng,
                reg_date,
                min(reg_date + timedelta(days=180), date(2022, 12, 31)),
            )

            accounts.append(
                Account(
                    account_id=account_id,
                    account_number=account_number,
                    customer_id=customer.customer_id,
                    bank_id=bank.bank_id,
                    account_type=acct_type,
                    balance=balance,
                    currency="USD",
                    opened_date=opened_date.isoformat(),
                    status="Active",
                    interest_rate=interest_rate,
                )
            )

    return accounts


def generate_portfolio_holdings(
    config: GeneratorConfig,
    rng: random.Random,
    accounts: list[Account],
    stocks: list[Stock],
    companies: list[Company],
    risk_profile_map: dict[str, str],
) -> list[PortfolioHolding]:
    """Generate holdings for investment accounts. Depends on accounts, stocks,
    companies, and the hidden risk-profile map so that stock selection and
    allocation weights correlate with each customer's risk profile.
    """
    holdings: list[PortfolioHolding] = []
    holding_counter = 0

    # Risk-aware holding count ranges
    _holdings_range: dict[str, tuple[int, int]] = {
        "Aggressive": (2, 3),  # concentrated
        "Conservative": (3, 5),  # diversified
        "Moderate": (2, 4),  # default
    }

    # Pre-build lookups used by _select_stocks_by_risk
    sector_by_company = {c.company_id: c.sector for c in companies}
    rng_np = np.random.default_rng(rng.getrandbits(128))

    investment_accounts = [a for a in accounts if a.account_type == "Investment"]

    for account in investment_accounts:
        risk = risk_profile_map.get(account.customer_id, "Moderate")
        lo, hi = _holdings_range.get(
            risk,
            (
                config.min_holdings_per_account,
                config.max_holdings_per_account,
            ),
        )
        num_holdings = rng.randint(lo, hi)

        selected_stocks = _select_stocks_by_risk(
            rng_np,
            stocks,
            sector_by_company,
            num_holdings,
            risk,
            strength=config.sector_preference_strength,
        )

        # Allocation weights: Aggressive = skewed, Conservative = flat
        if risk == "Aggressive":
            raw_weights = [rng.uniform(1, 10) for _ in selected_stocks]
        elif risk == "Conservative":
            raw_weights = [rng.uniform(3, 5) for _ in selected_stocks]
        else:
            raw_weights = [rng.uniform(1, 5) for _ in selected_stocks]
        total_weight = sum(raw_weights)

        for j, stock in enumerate(selected_stocks):
            holding_counter += 1
            holding_id = f"H{holding_counter:05d}"

            pct = round(raw_weights[j] / total_weight * 100, 1)

            # Purchase price: +/- 20% of current price
            purchase_price = round(stock.current_price * rng.uniform(0.80, 1.20), 2)

            # Purchase date: within the last 3 years
            purchase_date = _random_date(
                rng,
                date(2019, 1, 1),
                date(2022, 12, 31),
            )

            # Shares based on portfolio allocation and account balance
            allocation_value = account.balance * (pct / 100)
            shares = max(1, round(allocation_value / stock.current_price))

            current_value = round(shares * stock.current_price, 2)

            holdings.append(
                PortfolioHolding(
                    holding_id=holding_id,
                    account_id=account.account_id,
                    stock_id=stock.stock_id,
                    shares=shares,
                    purchase_price=purchase_price,
                    purchase_date=purchase_date.isoformat(),
                    current_value=current_value,
                    percentage_of_portfolio=pct,
                )
            )

    return holdings


def generate_transactions(
    config: GeneratorConfig,
    rng: random.Random,
    accounts: list[Account],
    risk_profile_map: dict[str, str],
) -> list[Transaction]:
    """Generate transactions between accounts. Depends on accounts.

    Creates a connected graph: most transactions go between random
    account pairs, with some internal transfers within the same bank.
    Transaction frequency correlates with risk profile:
    - Aggressive: +2 to max (more active trading)
    - Conservative: -1 to min and max (less active)
    - Moderate: default range
    """
    transactions: list[Transaction] = []
    txn_counter = 0

    account_ids = [a.account_id for a in accounts]

    # Build a bank-to-accounts index for internal transfers
    bank_accounts: dict[str, list[str]] = {}
    for a in accounts:
        bank_accounts.setdefault(a.bank_id, []).append(a.account_id)

    for account in accounts:
        risk = risk_profile_map.get(account.customer_id, "Moderate")
        min_txns = config.min_transactions_per_account
        max_txns = config.max_transactions_per_account
        if risk == "Aggressive":
            max_txns += 2
        elif risk == "Conservative":
            min_txns = max(1, min_txns - 1)
            max_txns = max(min_txns, max_txns - 1)
        num_txns = rng.randint(min_txns, max_txns)

        # Pre-build candidate lists for this account (avoids rebuilding per txn)
        other_accounts = [a for a in account_ids if a != account.account_id]
        same_bank_others = [
            a for a in bank_accounts.get(account.bank_id, []) if a != account.account_id
        ]

        for _ in range(num_txns):
            txn_counter += 1
            transaction_id = f"T{txn_counter:06d}"

            # 30% chance of internal transfer (same bank)
            if rng.random() < 0.30 and same_bank_others:
                to_account_id = rng.choice(same_bank_others)
            else:
                to_account_id = rng.choice(other_accounts)

            # Amount: log-normal
            amount = round(
                _log_normal_sample(
                    rng,
                    config.transaction_amount_median,
                    config.transaction_amount_sigma,
                    config.transaction_amount_min,
                    config.transaction_amount_max,
                ),
                2,
            )

            txn_type = rng.choices(
                TRANSACTION_TYPES,
                weights=TRANSACTION_TYPE_WEIGHTS,
            )[0]

            # Date: spread over past year (2023)
            txn_date = _random_date(rng, date(2023, 1, 1), date(2023, 12, 31))
            txn_hour = rng.randint(8, 17)
            txn_min = rng.randint(0, 59)
            txn_sec = rng.randint(0, 59)
            txn_time = f"{txn_hour:02d}:{txn_min:02d}:{txn_sec:02d}"

            description = rng.choice(TRANSACTION_DESCRIPTIONS)

            transactions.append(
                Transaction(
                    transaction_id=transaction_id,
                    from_account_id=account.account_id,
                    to_account_id=to_account_id,
                    amount=amount,
                    currency="USD",
                    transaction_date=txn_date.isoformat(),
                    transaction_time=txn_time,
                    type=txn_type,
                    status="Completed",
                    description=description,
                )
            )

    return transactions


def generate_fraud_rings(
    config: GeneratorConfig,
    rng: random.Random,
    customers: list[Customer],
    accounts: list[Account],
    banks: list[Bank],
    stocks: list[Stock],
    companies: list[Company],
    holdings: list[PortfolioHolding],
    transactions: list[Transaction],
    risk_profile_map: dict[str, str],
) -> None:
    """Mutate the entity lists in-place to add fraud ring patterns.

    For each ring:
    1. Create new customers with is_fraudulent=True
    2. Create checking + investment accounts at the same bank
    3. Generate circular transactions with structuring amounts ($8k-$9.999k)
    4. Add cross-ring transactions for density
    5. Give all ring members identical stock positions (coordinated pump-and-dump)
    """
    fake = _create_faker(config.random_seed + 99)

    # Determine starting IDs from existing data
    next_customer_num = len(customers) + 1
    next_account_num = len(accounts) + 1
    next_txn_num = len(transactions) + 1
    next_holding_num = len(holdings) + 1
    account_number_counter = 1001000000 + len(accounts) + 1

    # Pick 2-3 penny stocks (low market cap) for coordinated positions
    sorted_by_mcap = sorted(stocks, key=lambda s: s.market_cap_billions)
    penny_stocks = sorted_by_mcap[:3]

    for ring_idx in range(config.num_fraud_rings):
        # Pick one bank for the entire ring
        ring_bank = rng.choice(banks)

        ring_customers: list[Customer] = []
        ring_checking: list[Account] = []
        ring_investment: list[Account] = []

        # --- Step 1: Create fraud customers ---
        for i in range(config.fraud_ring_size):
            cid = f"C{next_customer_num:04d}"
            next_customer_num += 1

            first_name = fake.first_name()
            last_name = fake.last_name()

            customer = Customer(
                customer_id=cid,
                first_name=first_name,
                last_name=last_name,
                email=f"{first_name.lower()}.{last_name.lower()}@email.com",
                phone=f"555-{rng.randint(100, 999):03d}-{rng.randint(1000, 9999):04d}",
                address=fake.street_address(),
                city=fake.city(),
                state=rng.choice(US_STATES),
                zip_code=f"{rng.randint(10000, 99999)}",
                registration_date=_random_date(
                    rng, date(2022, 6, 1), date(2022, 9, 30)
                ).isoformat(),
                date_of_birth=_random_date(
                    rng, date(1975, 1, 1), date(1995, 12, 31)
                ).isoformat(),
                risk_profile="",
                employment_status=rng.choice(["Employed", "Self-Employed"]),
                annual_income=round(rng.uniform(50_000, 90_000)),
                credit_score=rng.randint(580, 680),
                is_fraudulent=True,
            )
            ring_customers.append(customer)
            customers.append(customer)
            risk_profile_map[cid] = "Aggressive"

        # --- Step 2: Create accounts (Checking + Investment) at the same bank ---
        for customer in ring_customers:
            for acct_type in ["Checking", "Investment"]:
                aid = f"A{next_account_num:05d}"
                next_account_num += 1
                account_number_counter += 1

                if acct_type == "Checking":
                    balance = round(rng.uniform(15_000, 50_000), 2)
                    interest_rate = 0.05
                else:
                    balance = round(rng.uniform(80_000, 200_000), 2)
                    interest_rate = 0.00

                reg_date = date.fromisoformat(customer.registration_date)
                opened_date = _random_date(
                    rng,
                    reg_date,
                    min(reg_date + timedelta(days=30), date(2022, 12, 31)),
                )

                acct = Account(
                    account_id=aid,
                    account_number=str(account_number_counter),
                    customer_id=customer.customer_id,
                    bank_id=ring_bank.bank_id,
                    account_type=acct_type,
                    balance=balance,
                    currency="USD",
                    opened_date=opened_date.isoformat(),
                    status="Active",
                    interest_rate=interest_rate,
                )
                accounts.append(acct)
                if acct_type == "Checking":
                    ring_checking.append(acct)
                else:
                    ring_investment.append(acct)

        # --- Step 3: Circular transactions (A→B→C→D→E→A) ---
        # Transactions happen on checking accounts, clustered in a 2-week burst
        burst_start = date(2023, 10, 1)
        burst_end = date(2023, 10, 14)

        for i in range(len(ring_checking)):
            from_acct = ring_checking[i]
            to_acct = ring_checking[(i + 1) % len(ring_checking)]

            for _ in range(config.fraud_transactions_per_pair):
                tid = f"T{next_txn_num:06d}"
                next_txn_num += 1

                amount = round(
                    rng.uniform(config.fraud_amount_min, config.fraud_amount_max), 2
                )
                txn_date = _random_date(rng, burst_start, burst_end)
                txn_hour = rng.randint(8, 23)
                txn_time = (
                    f"{txn_hour:02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"
                )

                transactions.append(
                    Transaction(
                        transaction_id=tid,
                        from_account_id=from_acct.account_id,
                        to_account_id=to_acct.account_id,
                        amount=amount,
                        currency="USD",
                        transaction_date=txn_date.isoformat(),
                        transaction_time=txn_time,
                        type="Transfer",
                        status="Completed",
                        description="Investment transfer",
                    )
                )

        # --- Step 4: Cross-ring transactions (non-adjacent pairs) ---
        for i in range(len(ring_checking)):
            for j in range(i + 2, len(ring_checking)):
                if j == (i - 1) % len(ring_checking):
                    continue  # skip adjacent (already covered)
                # 2 cross-ring transactions per non-adjacent pair
                for _ in range(2):
                    tid = f"T{next_txn_num:06d}"
                    next_txn_num += 1

                    amount = round(
                        rng.uniform(config.fraud_amount_min, config.fraud_amount_max), 2
                    )
                    txn_date = _random_date(rng, burst_start, burst_end)
                    txn_time = (
                        f"{rng.randint(8, 23):02d}:"
                        f"{rng.randint(0, 59):02d}:"
                        f"{rng.randint(0, 59):02d}"
                    )

                    transactions.append(
                        Transaction(
                            transaction_id=tid,
                            from_account_id=ring_checking[i].account_id,
                            to_account_id=ring_checking[j].account_id,
                            amount=amount,
                            currency="USD",
                            transaction_date=txn_date.isoformat(),
                            transaction_time=txn_time,
                            type="Transfer",
                            status="Completed",
                            description="Contract payment",
                        )
                    )

        # --- Step 5: Coordinated positions (all hold same penny stocks) ---
        for inv_acct in ring_investment:
            total_weight = sum(range(1, len(penny_stocks) + 1))
            for rank, stock in enumerate(penny_stocks):
                next_holding_num += 1
                hid = f"H{next_holding_num:05d}"

                pct = round((len(penny_stocks) - rank) / total_weight * 100, 1)
                purchase_price = round(stock.current_price * rng.uniform(0.85, 1.10), 2)
                purchase_date = _random_date(rng, date(2023, 9, 15), date(2023, 9, 30))
                allocation_value = inv_acct.balance * (pct / 100)
                shares = max(1, round(allocation_value / stock.current_price))
                current_value = round(shares * stock.current_price, 2)

                holdings.append(
                    PortfolioHolding(
                        holding_id=hid,
                        account_id=inv_acct.account_id,
                        stock_id=stock.stock_id,
                        shares=shares,
                        purchase_price=purchase_price,
                        purchase_date=purchase_date.isoformat(),
                        current_value=current_value,
                        percentage_of_portfolio=pct,
                    )
                )
