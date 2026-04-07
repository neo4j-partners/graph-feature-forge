"""Create Delta tables from CSV files on a UC volume.

Reads 7 CSV files from a UC volume, applies type transformations,
and creates 14 Delta tables (7 node tables, 7 relationship tables)
matching the Neo4j Spark Connector export format expected by
seeding.py and structured_data.py.

The relationship tables use ``source.{key}`` and ``target.{key}``
column naming to match the connector's ``relationship.nodes.map=false``
convention.
"""

from __future__ import annotations

from semantic_auth.structured_data import SQLExecutor

# ---------------------------------------------------------------------------
# Node table definitions
# ---------------------------------------------------------------------------

#: CSV filename -> Delta table name
NODE_CSV_TABLES: dict[str, str] = {
    "customers.csv": "customer",
    "banks.csv": "bank",
    "accounts.csv": "account",
    "companies.csv": "company",
    "stocks.csv": "stock",
    "portfolio_holdings.csv": "position",
    "transactions.csv": "transaction",
}

#: Per-table SQL SELECT with type casts.  Keyed by Delta table name.
#: The ``{volume_path}`` placeholder is filled at runtime.
_NODE_SQL: dict[str, str] = {
    "customer": """\
SELECT
    customer_id,
    first_name,
    last_name,
    email,
    phone,
    address,
    city,
    state,
    zip_code,
    CAST(registration_date AS DATE) AS registration_date,
    CAST(date_of_birth AS DATE) AS date_of_birth,
    risk_profile,
    employment_status,
    CAST(annual_income AS INT) AS annual_income,
    CAST(credit_score AS INT) AS credit_score
FROM read_files(
    '{volume_path}/csv/customers.csv',
    format => 'csv', header => true
)""",
    "bank": """\
SELECT
    bank_id,
    name,
    headquarters,
    bank_type,
    CAST(total_assets_billions AS DOUBLE) AS total_assets_billions,
    CAST(established_year AS INT) AS established_year,
    routing_number,
    swift_code
FROM read_files(
    '{volume_path}/csv/banks.csv',
    format => 'csv', header => true
)""",
    "account": """\
SELECT
    account_id,
    account_number,
    customer_id,
    bank_id,
    account_type,
    CAST(balance AS DOUBLE) AS balance,
    currency,
    CAST(opened_date AS DATE) AS opened_date,
    status,
    CAST(interest_rate AS DOUBLE) AS interest_rate
FROM read_files(
    '{volume_path}/csv/accounts.csv',
    format => 'csv', header => true
)""",
    "company": """\
SELECT
    company_id,
    name,
    ticker_symbol,
    industry,
    sector,
    CAST(market_cap_billions AS DOUBLE) AS market_cap_billions,
    headquarters,
    CAST(founded_year AS INT) AS founded_year,
    ceo,
    CAST(employee_count AS INT) AS employee_count,
    CAST(annual_revenue_billions AS DOUBLE) AS annual_revenue_billions
FROM read_files(
    '{volume_path}/csv/companies.csv',
    format => 'csv', header => true
)""",
    "stock": """\
SELECT
    stock_id,
    ticker,
    company_id,
    CAST(current_price AS DOUBLE) AS current_price,
    CAST(previous_close AS DOUBLE) AS previous_close,
    CAST(opening_price AS DOUBLE) AS opening_price,
    CAST(day_high AS DOUBLE) AS day_high,
    CAST(day_low AS DOUBLE) AS day_low,
    CAST(volume AS INT) AS volume,
    CAST(market_cap_billions AS DOUBLE) AS market_cap_billions,
    CAST(pe_ratio AS DOUBLE) AS pe_ratio,
    CAST(dividend_yield AS DOUBLE) AS dividend_yield,
    CAST(fifty_two_week_high AS DOUBLE) AS fifty_two_week_high,
    CAST(fifty_two_week_low AS DOUBLE) AS fifty_two_week_low,
    exchange
FROM read_files(
    '{volume_path}/csv/stocks.csv',
    format => 'csv', header => true
)""",
    "position": """\
SELECT
    holding_id AS position_id,
    account_id,
    stock_id,
    CAST(shares AS INT) AS shares,
    CAST(purchase_price AS DOUBLE) AS purchase_price,
    CAST(purchase_date AS DATE) AS purchase_date,
    CAST(current_value AS DOUBLE) AS current_value,
    CAST(percentage_of_portfolio AS DOUBLE) AS percentage_of_portfolio
FROM read_files(
    '{volume_path}/csv/portfolio_holdings.csv',
    format => 'csv', header => true
)""",
    "transaction": """\
SELECT
    transaction_id,
    from_account_id,
    to_account_id,
    CAST(amount AS DOUBLE) AS amount,
    currency,
    CAST(transaction_date AS DATE) AS transaction_date,
    transaction_time,
    type,
    status,
    description
FROM read_files(
    '{volume_path}/csv/transactions.csv',
    format => 'csv', header => true
)""",
}

# ---------------------------------------------------------------------------
# Relationship table definitions
# ---------------------------------------------------------------------------

#: Relationship table -> (source_table, source_col, source_alias, target_col, target_alias)
#: source_alias/target_alias are the column names in the relationship table
#: (the ``source.{key}`` / ``target.{key}`` convention).
RELATIONSHIP_DEFS: dict[str, tuple[str, str, str, str, str]] = {
    "has_account": (
        "account",
        "customer_id",
        "customer_id",
        "account_id",
        "account_id",
    ),
    "at_bank": ("account", "account_id", "account_id", "bank_id", "bank_id"),
    "of_company": ("stock", "stock_id", "stock_id", "company_id", "company_id"),
    "performs": (
        "transaction",
        "from_account_id",
        "account_id",
        "transaction_id",
        "transaction_id",
    ),
    "benefits_to": (
        "transaction",
        "transaction_id",
        "transaction_id",
        "to_account_id",
        "account_id",
    ),
    "has_position": (
        "position",
        "account_id",
        "account_id",
        "position_id",
        "position_id",
    ),
    "of_security": ("position", "position_id", "position_id", "stock_id", "stock_id"),
}


def _node_table_sql(
    table_name: str, catalog: str, schema: str, volume_path: str
) -> str:
    """Return the CREATE TABLE SQL for a node table."""
    select = _NODE_SQL[table_name].replace("{volume_path}", volume_path)
    return f"CREATE OR REPLACE TABLE `{catalog}`.`{schema}`.`{table_name}` AS\n{select}"


def _relationship_table_sql(rel_name: str, catalog: str, schema: str) -> str:
    """Return the CREATE TABLE SQL for a relationship table."""
    src_table, src_col, src_alias, tgt_col, tgt_alias = RELATIONSHIP_DEFS[rel_name]
    return (
        f"CREATE OR REPLACE TABLE `{catalog}`.`{schema}`.`{rel_name}` AS\n"
        f"SELECT\n"
        f"    {src_col} AS `source.{src_alias}`,\n"
        f"    {tgt_col} AS `target.{tgt_alias}`\n"
        f"FROM `{catalog}`.`{schema}`.`{src_table}`"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_node_tables(
    execute_sql: SQLExecutor,
    catalog: str,
    schema: str,
    volume_path: str,
) -> dict[str, int]:
    """Create 7 node Delta tables from CSV files on the volume.

    Returns ``{table_name: row_count}``.
    """
    counts: dict[str, int] = {}
    for csv_file, table_name in NODE_CSV_TABLES.items():
        sql = _node_table_sql(table_name, catalog, schema, volume_path)
        execute_sql(sql)

        rows = execute_sql(
            f"SELECT COUNT(*) AS cnt FROM `{catalog}`.`{schema}`.`{table_name}`"
        )
        count = rows[0]["cnt"] if rows else 0
        counts[table_name] = count
        print(f"    {table_name}: {count} rows (from {csv_file})")

    return counts


def create_relationship_tables(
    execute_sql: SQLExecutor,
    catalog: str,
    schema: str,
) -> dict[str, int]:
    """Create 7 relationship Delta tables derived from node tables.

    Each table has two columns using the ``source.{key}`` /
    ``target.{key}`` naming convention.

    Returns ``{table_name: row_count}``.
    """
    counts: dict[str, int] = {}
    for rel_name in RELATIONSHIP_DEFS:
        sql = _relationship_table_sql(rel_name, catalog, schema)
        execute_sql(sql)

        rows = execute_sql(
            f"SELECT COUNT(*) AS cnt FROM `{catalog}`.`{schema}`.`{rel_name}`"
        )
        count = rows[0]["cnt"] if rows else 0
        counts[rel_name] = count
        print(f"    {rel_name}: {count} rows")

    return counts


def load_all(
    execute_sql: SQLExecutor,
    catalog: str,
    schema: str,
    volume_path: str,
) -> dict[str, int]:
    """Create all 14 Delta tables from CSV files.

    Creates the schema if it doesn't exist, then creates 7 node tables
    from CSVs on the volume, then derives 7 relationship tables.

    Returns ``{table_name: row_count}``.
    """
    counts: dict[str, int] = {}

    execute_sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")

    print("\n  Creating node tables ...")
    node_counts = create_node_tables(execute_sql, catalog, schema, volume_path)
    counts.update(node_counts)

    print("\n  Creating relationship tables ...")
    rel_counts = create_relationship_tables(execute_sql, catalog, schema)
    counts.update(rel_counts)

    return counts
