"""Configuration for the financial portfolio data generator.

All counts and distribution parameters are configurable via environment
variables or a .env file. Defaults match the target scale from grow.md.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class GeneratorConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="GEN_",
    )

    # --- Entity counts ---
    num_customers: int = 500
    num_banks: int = 50
    num_companies: int = 200

    # --- Labeling ---
    labeled_fraction: float = 0.30
    labels_per_class: int = 100  # Aggressive, Conservative, Moderate

    # --- Accounts per customer distribution (weights for 1, 2, 3 accounts) ---
    accounts_weight_one: float = 0.40
    accounts_weight_two: float = 0.40
    accounts_weight_three: float = 0.20

    # --- Holdings per investment account ---
    min_holdings_per_account: int = 2
    max_holdings_per_account: int = 5

    # --- Risk-aware portfolio settings ---
    # Controls how strongly stock selection correlates with risk profile.
    # 0.0 = purely random (backward-compatible), 1.0 = default bias, 2.0 = strong bias.
    sector_preference_strength: float = 1.8

    # --- Transactions per account ---
    min_transactions_per_account: int = 3
    max_transactions_per_account: int = 5

    # --- Customer distributions ---
    income_median: float = 75_000.0
    income_sigma: float = 0.6  # log-normal sigma
    income_min: float = 30_000.0
    income_max: float = 500_000.0

    credit_score_mean: float = 700.0
    credit_score_std: float = 80.0
    credit_score_min: int = 300
    credit_score_max: int = 850

    # --- Transaction amounts ---
    transaction_amount_median: float = 500.0
    transaction_amount_sigma: float = 1.2
    transaction_amount_min: float = 10.0
    transaction_amount_max: float = 50_000.0

    # --- Bank assets ---
    bank_assets_min_billions: float = 1.0
    bank_assets_max_billions: float = 500.0

    # --- Fraud ring settings ---
    num_fraud_rings: int = 1
    fraud_ring_size: int = 5  # customers per ring
    fraud_transactions_per_pair: int = 8  # circular txns between each adjacent pair
    fraud_amount_min: float = 8_000.0  # structuring: just under $10k
    fraud_amount_max: float = 9_999.0

    # --- Generator settings ---
    random_seed: int = 42
    output_dir: str = "csv"
