"""Structured data access for gap analysis.

Executes SQL queries against the workshop's 14 Delta tables exported
from Neo4j by Lab 4 and formats results as context strings for LLM
synthesis.

Tables live in the workshop catalog (default: neo4j_augmentation_demo).
Node tables have standard column names (customer_id, first_name, etc.).
Relationship tables have source/target node properties prefixed with
``source.`` and ``target.`` — the Neo4j Spark Connector convention when
``relationship.nodes.map=false``.

Node tables (7):
    customer, bank, account, company, stock, position, transaction

Relationship tables (7):
    has_account (Customer->Account), at_bank (Account->Bank),
    of_company (Stock->Company), performs (Account->Transaction),
    benefits_to (Transaction->Account), has_position (Account->Position),
    of_security (Position->Stock)
"""

from __future__ import annotations

from typing import Any, Callable

QueryResult = list[dict[str, Any]]
SQLExecutor = Callable[[str], QueryResult]

# All 14 tables from the Lab 4 export
NODE_TABLES = ["customer", "bank", "account", "company", "stock", "position", "transaction"]
RELATIONSHIP_TABLES = [
    "has_account", "at_bank", "of_company", "performs",
    "benefits_to", "has_position", "of_security",
]


# ---------------------------------------------------------------------------
# SQL executor factories
# ---------------------------------------------------------------------------


def make_spark_executor(spark_session: Any = None) -> SQLExecutor:
    """Create an executor using a Spark session.

    On serverless tasks, PySpark is pre-installed so ``SparkSession``
    is tried first.  Falls back to ``DatabricksSession`` (databricks-connect)
    for local development.  Pass an explicit session to skip auto-detection.
    """
    if spark_session is None:
        # Try PySpark first (available on serverless and cluster tasks).
        try:
            from pyspark.sql import SparkSession
            spark_session = SparkSession.builder.getOrCreate()
        except Exception:
            pass

        # Fall back to Databricks Connect (local development).
        if spark_session is None:
            try:
                from databricks.connect import DatabricksSession
                spark_session = DatabricksSession.builder.getOrCreate()
            except Exception as exc:
                raise RuntimeError(
                    "No Spark session available. Pass one explicitly, "
                    "provide --warehouse-id for SDK execution, or "
                    "run on a Databricks cluster."
                ) from exc

    def execute(query: str) -> QueryResult:
        return [row.asDict() for row in spark_session.sql(query).collect()]

    return execute


def make_sdk_executor(warehouse_id: str) -> SQLExecutor:
    """Create an executor using Databricks SDK statement execution.

    Works locally and on-cluster. Requires a running SQL warehouse.
    """
    from databricks.sdk import WorkspaceClient

    wc = WorkspaceClient()

    def execute(query: str) -> QueryResult:
        response = wc.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=query,
            wait_timeout="30s",
        )
        state = response.status.state.value
        if state != "SUCCEEDED":
            error = getattr(response.status, "error", None)
            msg = error.message if error else state
            raise RuntimeError(f"SQL execution failed: {msg}")
        cols = [c.name for c in response.manifest.schema.columns]
        return [dict(zip(cols, row)) for row in response.result.data_array]

    return execute


# ---------------------------------------------------------------------------
# Structured data access
# ---------------------------------------------------------------------------


class StructuredDataAccess:
    """Query workshop Delta tables and format results as LLM context.

    Args:
        execute_sql: Callable that takes a SQL string and returns rows
            as a list of dicts. Use ``make_spark_executor`` or
            ``make_sdk_executor`` to create one.
        catalog: Unity Catalog name containing the Delta tables.
        schema: Schema name within the catalog.
    """

    def __init__(
        self,
        execute_sql: SQLExecutor,
        catalog: str = "neo4j_augmentation_demo",
        schema: str = "raw_data",
    ) -> None:
        self._sql = execute_sql
        self._catalog = catalog
        self._schema = schema

    def _t(self, name: str) -> str:
        """Fully qualified, backtick-quoted table reference."""
        return f"`{self._catalog}`.`{self._schema}`.`{name}`"

    @staticmethod
    def _src(col: str) -> str:
        """Source-node column reference in a relationship table."""
        return f"`source.{col}`"

    @staticmethod
    def _tgt(col: str) -> str:
        """Target-node column reference in a relationship table."""
        return f"`target.{col}`"

    # -----------------------------------------------------------------
    # Schema discovery
    # -----------------------------------------------------------------

    def discover_schema(self) -> dict[str, list[str]]:
        """Return ``{table_name: [column_names]}`` for all 14 tables.

        Run this first to verify column names before running gap
        analysis queries. The relationship table columns depend on
        the Neo4j Spark Connector version used during the Lab 4 export.
        """
        schema_map: dict[str, list[str]] = {}
        for table in NODE_TABLES + RELATIONSHIP_TABLES:
            try:
                rows = self._sql(f"DESCRIBE TABLE {self._t(table)}")
                schema_map[table] = [r["col_name"] for r in rows]
            except Exception as exc:
                schema_map[table] = [f"ERROR: {exc}"]
        return schema_map

    # -----------------------------------------------------------------
    # Gap analysis context queries
    # -----------------------------------------------------------------

    def get_portfolio_holdings(self) -> str:
        """Customer portfolio holdings grouped by sector.

        Used for interest-holding gap analysis: what does each customer
        actually hold, so the synthesis LLM can compare against stated
        interests from the documents.
        """
        query = f"""
        SELECT
            c.customer_id,
            c.first_name,
            c.last_name,
            a.account_id,
            a.account_type,
            s.ticker,
            s.current_price,
            co.name       AS company_name,
            co.sector,
            p.shares,
            p.purchase_price,
            p.current_value,
            p.percentage_of_portfolio
        FROM {self._t('customer')} c
        JOIN {self._t('has_account')} ha
            ON c.customer_id = ha.{self._src('customer_id')}
        JOIN {self._t('account')} a
            ON ha.{self._tgt('account_id')} = a.account_id
        JOIN {self._t('has_position')} hp
            ON a.account_id = hp.{self._src('account_id')}
        JOIN {self._t('position')} p
            ON hp.{self._tgt('position_id')} = p.position_id
        JOIN {self._t('of_security')} os
            ON p.position_id = os.{self._src('position_id')}
        JOIN {self._t('stock')} s
            ON os.{self._tgt('stock_id')} = s.stock_id
        JOIN {self._t('of_company')} oc
            ON s.stock_id = oc.{self._src('stock_id')}
        JOIN {self._t('company')} co
            ON oc.{self._tgt('company_id')} = co.company_id
        ORDER BY c.customer_id, co.sector, s.ticker
        """
        rows = self._sql(query)
        return self._format_portfolio_holdings(rows)

    def get_customer_profiles(self) -> str:
        """Customer demographics and account summary.

        Used for risk profile alignment: annual income, credit score,
        risk profile, employment status, account types and balances,
        portfolio composition by sector.
        """
        customer_query = f"""
        SELECT
            c.customer_id,
            c.first_name,
            c.last_name,
            c.email,
            c.city,
            c.state,
            c.annual_income,
            c.credit_score,
            c.date_of_birth,
            c.registration_date,
            c.risk_profile,
            c.employment_status
        FROM {self._t('customer')} c
        ORDER BY c.customer_id
        """

        account_query = f"""
        SELECT
            ha.{self._src('customer_id')} AS customer_id,
            a.account_id,
            a.account_type,
            a.balance,
            a.status,
            a.interest_rate
        FROM {self._t('has_account')} ha
        JOIN {self._t('account')} a
            ON ha.{self._tgt('account_id')} = a.account_id
        ORDER BY customer_id, a.account_id
        """

        customers = self._sql(customer_query)
        accounts = self._sql(account_query)
        return self._format_customer_profiles(customers, accounts)

    def get_data_completeness(self) -> str:
        """Customer field completeness analysis.

        Used for data quality gap analysis: which customer fields are
        populated vs null, and which fields mentioned in profile
        documents (occupation, employer, investment philosophy) are
        absent from the schema entirely.
        """
        query = f"SELECT * FROM {self._t('customer')} ORDER BY customer_id"
        rows = self._sql(query)
        return self._format_data_completeness(rows)

    def get_all_structured_context(self) -> str:
        """Combined structured data context for comprehensive gap analysis.

        Concatenates portfolio holdings, customer profiles, and data
        completeness into a single context block with clear section
        headers.
        """
        sections = [
            self.get_portfolio_holdings(),
            self.get_customer_profiles(),
            self.get_data_completeness(),
        ]
        return "\n\n".join(sections)

    # -----------------------------------------------------------------
    # Formatting helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _format_portfolio_holdings(rows: QueryResult) -> str:
        """Group holdings by customer, then by sector."""
        lines = ["# Customer Portfolio Holdings", ""]

        if not rows:
            lines.append("No portfolio holdings data found.")
            return "\n".join(lines)

        # Group by customer
        customers: dict[str, list[dict]] = {}
        for r in rows:
            cid = r["customer_id"]
            customers.setdefault(cid, []).append(r)

        for cid, holdings in customers.items():
            first = holdings[0]
            name = f"{first['first_name']} {first['last_name']}"
            lines.append(f"## {name} ({cid})")

            # Group by sector
            sectors: dict[str, list[dict]] = {}
            for h in holdings:
                sector = h.get("sector", "Unknown")
                sectors.setdefault(sector, []).append(h)

            total_value = sum(_to_float(h.get("current_value", 0)) for h in holdings)
            lines.append(f"Total portfolio value: ${total_value:,.2f}")
            lines.append("")

            for sector, positions in sorted(sectors.items()):
                sector_value = sum(_to_float(p.get("current_value", 0)) for p in positions)
                sector_pct = (sector_value / total_value * 100) if total_value else 0
                lines.append(f"  {sector} ({sector_pct:.1f}% of portfolio):")
                for p in positions:
                    ticker = p.get("ticker", "?")
                    company = p.get("company_name", "?")
                    shares = p.get("shares", "?")
                    value = _to_float(p.get("current_value", 0))
                    pct = _to_float(p.get("percentage_of_portfolio", 0))
                    lines.append(
                        f"    {ticker} ({company}): "
                        f"{shares} shares, ${value:,.2f} value, "
                        f"{pct:.1f}% of portfolio"
                    )
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_customer_profiles(
        customers: QueryResult,
        accounts: QueryResult,
    ) -> str:
        """Format customer demographics with account summaries."""
        lines = ["# Customer Demographics and Accounts", ""]

        if not customers:
            lines.append("No customer data found.")
            return "\n".join(lines)

        # Index accounts by customer_id
        acct_map: dict[str, list[dict]] = {}
        for a in accounts:
            cid = a["customer_id"]
            acct_map.setdefault(cid, []).append(a)

        for c in customers:
            cid = c["customer_id"]
            name = f"{c['first_name']} {c['last_name']}"
            lines.append(f"## {name} ({cid})")

            income = c.get("annual_income")
            credit = c.get("credit_score")
            city = c.get("city", "")
            state = c.get("state", "")
            dob = c.get("date_of_birth", "")
            reg = c.get("registration_date", "")

            details = []
            if city or state:
                details.append(f"Location: {city}, {state}")
            if income is not None:
                details.append(f"Annual income: ${_to_float(income):,.0f}")
            if credit is not None:
                details.append(f"Credit score: {credit}")
            if dob:
                details.append(f"Date of birth: {dob}")
            if reg:
                details.append(f"Member since: {reg}")
            risk = c.get("risk_profile", "")
            if risk:
                details.append(f"Risk profile: {risk}")
            emp = c.get("employment_status", "")
            if emp:
                details.append(f"Employment: {emp}")
            lines.append("  " + " | ".join(details))

            cust_accounts = acct_map.get(cid, [])
            if cust_accounts:
                lines.append(f"  Accounts ({len(cust_accounts)}):")
                for a in cust_accounts:
                    atype = a.get("account_type", "?")
                    balance = _to_float(a.get("balance", 0))
                    status = a.get("status", "?")
                    lines.append(
                        f"    {a['account_id']} ({atype}): "
                        f"${balance:,.2f}, status={status}"
                    )
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_data_completeness(rows: QueryResult) -> str:
        """Analyze field coverage and identify schema gaps."""
        lines = [
            "# Customer Data Completeness Analysis",
            "",
            f"Total customers: {len(rows)}",
            "",
        ]

        if not rows:
            lines.append("No customer data found.")
            return "\n".join(lines)

        # Analyze field coverage
        all_fields = list(rows[0].keys())
        # Exclude Neo4j internal columns
        data_fields = [f for f in all_fields if not f.startswith("<")]

        lines.append("## Fields Present in Schema")
        lines.append("")
        for field in sorted(data_fields):
            non_null = sum(1 for r in rows if r.get(field) is not None and str(r.get(field, "")).strip() != "")
            pct = non_null / len(rows) * 100
            lines.append(f"  {field}: {non_null}/{len(rows)} populated ({pct:.0f}%)")

        lines.append("")
        lines.append("## Fields NOT in Schema (mentioned in customer profile documents)")
        lines.append("  These attributes appear in the HTML customer profiles but are")
        lines.append("  not captured as columns in the structured customer table:")
        lines.append("")
        missing_fields = [
            ("occupation", "Job title and role"),
            ("employer", "Company or organization"),
            ("investment_philosophy", "Investment approach and strategy"),
            ("life_stage", "Career/life stage (mid-career, pre-retirement, etc.)"),
            ("financial_goals", "Stated financial objectives"),
            ("communication_preference", "Preferred contact method"),
        ]
        for field_name, desc in missing_fields:
            lines.append(f"  {field_name}: {desc} — NOT IN SCHEMA")

        lines.append("")
        lines.append("## Sample Customer Records (first 5)")
        lines.append("")
        for r in rows[:5]:
            cid = r.get("customer_id", "?")
            fname = r.get("first_name", "?")
            lname = r.get("last_name", "?")
            email_flag = "yes" if r.get("email") else "no"
            phone_flag = "yes" if r.get("phone") else "no"
            income = r.get("annual_income", "?")
            credit = r.get("credit_score", "?")
            lines.append(
                f"  {cid} {fname} {lname}: "
                f"email={email_flag}, phone={phone_flag}, "
                f"income={income}, credit_score={credit}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> float:
    """Coerce a value to float, returning 0.0 on failure."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0
