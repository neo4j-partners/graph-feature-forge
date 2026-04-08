"""Shared graph metadata for node and relationship types.

Single source of truth for the structural metadata that ``loading.py``,
``seeding.py``, ``structured_data.py``, and ``extraction.py`` all need.
Centralises label-to-table mappings, key properties, and the sets of
base types used by selective extraction.

This module deliberately does **not** generate SQL — ``loading.py``
keeps its hand-tuned ``_NODE_SQL`` templates because they contain
CSV-specific CASTs and column renames that don't belong in a generic
schema registry.
"""

from __future__ import annotations

from typing import NamedTuple

# ---------------------------------------------------------------------------
# Typed metadata containers
# ---------------------------------------------------------------------------


class NodeMeta(NamedTuple):
    """Metadata for a base node type."""

    key_property: str  # e.g. "customer_id"
    delta_table: str  # e.g. "customer"


class RelationshipMeta(NamedTuple):
    """Metadata for a base relationship type."""

    source_label: str  # e.g. "Customer"
    source_key: str  # e.g. "customer_id"
    target_label: str  # e.g. "Account"
    target_key: str  # e.g. "account_id"
    delta_table: str  # e.g. "has_account"


# NOTE: enrichment.md proposes richer PropertySchema / NodeSchema /
# RelationshipSchema dataclasses with per-column Delta types and CSV
# column mappings.  The simpler NamedTuples above are a deliberate
# choice: at current scale, ``overwriteSchema=true`` handles type
# evolution in extraction, and ``loading.py`` keeps hand-tuned SQL
# CASTs with CSV-specific column renames that are not worth
# abstracting into a generic registry.

# ---------------------------------------------------------------------------
# Node metadata
# ---------------------------------------------------------------------------

#: Neo4j label -> NodeMeta
NODE_TABLES: dict[str, NodeMeta] = {
    "Customer": NodeMeta("customer_id", "customer"),
    "Bank": NodeMeta("bank_id", "bank"),
    "Account": NodeMeta("account_id", "account"),
    "Company": NodeMeta("company_id", "company"),
    "Stock": NodeMeta("stock_id", "stock"),
    "Position": NodeMeta("position_id", "position"),
    "Transaction": NodeMeta("transaction_id", "transaction"),
}

# ---------------------------------------------------------------------------
# Relationship metadata
# ---------------------------------------------------------------------------

#: Neo4j relationship type -> RelationshipMeta
RELATIONSHIP_TABLES: dict[str, RelationshipMeta] = {
    "HAS_ACCOUNT": RelationshipMeta("Customer", "customer_id", "Account", "account_id", "has_account"),
    "AT_BANK": RelationshipMeta("Account", "account_id", "Bank", "bank_id", "at_bank"),
    "OF_COMPANY": RelationshipMeta("Stock", "stock_id", "Company", "company_id", "of_company"),
    "PERFORMS": RelationshipMeta("Account", "account_id", "Transaction", "transaction_id", "performs"),
    "BENEFITS_TO": RelationshipMeta("Transaction", "transaction_id", "Account", "account_id", "benefits_to"),
    "HAS_POSITION": RelationshipMeta("Account", "account_id", "Position", "position_id", "has_position"),
    "OF_SECURITY": RelationshipMeta("Position", "position_id", "Stock", "stock_id", "of_security"),
}

# ---------------------------------------------------------------------------
# CSV mapping (used only by loading.py)
# ---------------------------------------------------------------------------

#: CSV filename -> delta_table_name
NODE_CSV_FILES: dict[str, str] = {
    "customers.csv": "customer",
    "banks.csv": "bank",
    "accounts.csv": "account",
    "companies.csv": "company",
    "stocks.csv": "stock",
    "portfolio_holdings.csv": "position",
    "transactions.csv": "transaction",
}

# ---------------------------------------------------------------------------
# Base label / type sets (for selective extraction)
# ---------------------------------------------------------------------------

#: Node labels that come from CSV loading — extraction should skip these.
BASE_NODE_LABELS: frozenset[str] = frozenset(NODE_TABLES.keys())

#: Relationship types that come from CSV loading — extraction should skip these.
BASE_RELATIONSHIP_TYPES: frozenset[str] = frozenset(RELATIONSHIP_TABLES.keys())

# ---------------------------------------------------------------------------
# Derived convenience lists
# ---------------------------------------------------------------------------

#: Delta table names for all base node tables (sorted by label).
NODE_TABLE_NAMES: list[str] = sorted(m.delta_table for m in NODE_TABLES.values())

#: Delta table names for all base relationship tables (sorted by type).
RELATIONSHIP_TABLE_NAMES: list[str] = sorted(m.delta_table for m in RELATIONSHIP_TABLES.values())
