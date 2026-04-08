"""Orchestrates data generation and writes CSVs to the output directory.

Usage:
    python -m data_generator.main
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd

from .config import GeneratorConfig
from .generators import (
    generate_accounts,
    generate_banks,
    generate_companies,
    generate_customers,
    generate_portfolio_holdings,
    generate_stocks,
    generate_transactions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _write_csv(path: Path, rows: list) -> None:
    """Write a list of Pydantic models to a CSV file.

    Column order is determined by the model's field definitions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row.model_dump() for row in rows])
    df.to_csv(path, index=False)
    logger.info("Wrote %d rows to %s", len(rows), path)


def main() -> None:
    logger.info("Loading configuration...")
    config = GeneratorConfig()

    rng = random.Random(config.random_seed)
    output_dir = Path(config.output_dir)

    logger.info("Configuration:")
    logger.info("  Customers:  %d", config.num_customers)
    logger.info("  Banks:      %d", config.num_banks)
    logger.info("  Companies:  %d", config.num_companies)
    logger.info("  Seed:       %d", config.random_seed)
    logger.info("  Output:     %s", output_dir)

    # --- Generate in dependency order ---

    logger.info("Generating banks...")
    banks = generate_banks(config, rng)

    logger.info("Generating companies...")
    companies = generate_companies(config, rng)

    logger.info("Generating stocks...")
    stocks = generate_stocks(config, rng, companies)

    logger.info("Generating customers...")
    customers, risk_profile_map = generate_customers(config, rng)

    logger.info("Generating accounts...")
    accounts = generate_accounts(config, rng, customers, banks)

    logger.info("Generating portfolio holdings...")
    holdings = generate_portfolio_holdings(
        config, rng, accounts, stocks, companies, risk_profile_map,
    )

    logger.info("Generating transactions...")
    transactions = generate_transactions(config, rng, accounts, risk_profile_map)

    # --- Write CSVs ---

    logger.info("Writing CSV files to %s ...", output_dir)

    _write_csv(output_dir / "banks.csv", banks)
    _write_csv(output_dir / "companies.csv", companies)
    _write_csv(output_dir / "stocks.csv", stocks)
    _write_csv(output_dir / "customers.csv", customers)
    _write_csv(output_dir / "accounts.csv", accounts)
    _write_csv(output_dir / "portfolio_holdings.csv", holdings)
    _write_csv(output_dir / "transactions.csv", transactions)

    # --- Summary ---

    labeled_count = sum(1 for c in customers if c.risk_profile)
    investment_count = sum(1 for a in accounts if a.account_type == "Investment")

    logger.info("--- Generation Summary ---")
    logger.info("  Banks:              %d", len(banks))
    logger.info("  Companies:          %d", len(companies))
    logger.info("  Stocks:             %d", len(stocks))
    logger.info("  Customers:          %d (labeled: %d)", len(customers), labeled_count)
    logger.info("  Accounts:           %d (investment: %d)", len(accounts), investment_count)
    logger.info("  Portfolio holdings:  %d", len(holdings))
    logger.info("  Transactions:       %d", len(transactions))
    logger.info("Done.")


def __main__() -> None:
    main()


if __name__ == "__main__":
    main()
