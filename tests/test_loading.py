"""Tests for loading module — verifies SQL generation."""

from semantic_auth.graph_schema import NODE_CSV_FILES, RELATIONSHIP_TABLES
from semantic_auth.loading import (
    RELATIONSHIP_DEFS,
    _NODE_SQL,
    _node_table_sql,
    _relationship_table_sql,
)


class TestNodeCsvTables:
    def test_all_seven_csvs_mapped(self):
        assert len(NODE_CSV_FILES) == 7

    def test_position_maps_from_portfolio_holdings(self):
        assert NODE_CSV_FILES["portfolio_holdings.csv"] == "position"

    def test_all_sql_templates_exist(self):
        for table_name in NODE_CSV_FILES.values():
            assert table_name in _NODE_SQL, f"Missing SQL for {table_name}"


class TestNodeTableSql:
    def test_customer_type_casts(self):
        sql = _node_table_sql("customer", "cat", "sch", "/Volumes/cat/sch/vol")
        assert "CAST(annual_income AS INT)" in sql
        assert "CAST(credit_score AS INT)" in sql
        assert "CAST(registration_date AS DATE)" in sql
        assert "CAST(date_of_birth AS DATE)" in sql

    def test_position_renames_holding_id(self):
        sql = _node_table_sql("position", "cat", "sch", "/Volumes/cat/sch/vol")
        assert "holding_id AS position_id" in sql

    def test_account_type_casts(self):
        sql = _node_table_sql("account", "cat", "sch", "/Volumes/cat/sch/vol")
        assert "CAST(balance AS DOUBLE)" in sql
        assert "CAST(interest_rate AS DOUBLE)" in sql
        assert "CAST(opened_date AS DATE)" in sql

    def test_stock_type_casts(self):
        sql = _node_table_sql("stock", "cat", "sch", "/Volumes/cat/sch/vol")
        assert "CAST(current_price AS DOUBLE)" in sql
        assert "CAST(volume AS INT)" in sql

    def test_volume_path_substituted(self):
        sql = _node_table_sql("customer", "cat", "sch", "/Volumes/my-cat/my-sch/my-vol")
        assert "/Volumes/my-cat/my-sch/my-vol/csv/customers.csv" in sql
        assert "{volume_path}" not in sql

    def test_creates_table(self):
        sql = _node_table_sql("bank", "mycat", "mysch", "/Volumes/cat/sch/vol")
        assert sql.startswith("CREATE OR REPLACE TABLE `mycat`.`mysch`.`bank`")


class TestRelationshipDefs:
    def test_all_seven_defined(self):
        assert len(RELATIONSHIP_DEFS) == 7

    def test_all_relationships_present(self):
        expected = {
            "has_account",
            "at_bank",
            "of_company",
            "performs",
            "benefits_to",
            "has_position",
            "of_security",
        }
        assert set(RELATIONSHIP_DEFS.keys()) == expected


class TestRelationshipTableSql:
    def test_has_account_columns(self):
        sql = _relationship_table_sql("has_account", "cat", "sch")
        assert "`source.customer_id`" in sql
        assert "`target.account_id`" in sql
        assert "FROM `cat`.`sch`.`account`" in sql

    def test_at_bank_columns(self):
        sql = _relationship_table_sql("at_bank", "cat", "sch")
        assert "`source.account_id`" in sql
        assert "`target.bank_id`" in sql

    def test_of_company_columns(self):
        sql = _relationship_table_sql("of_company", "cat", "sch")
        assert "`source.stock_id`" in sql
        assert "`target.company_id`" in sql

    def test_performs_renames_from_account_id(self):
        sql = _relationship_table_sql("performs", "cat", "sch")
        assert "from_account_id AS `source.account_id`" in sql
        assert "transaction_id AS `target.transaction_id`" in sql

    def test_benefits_to_renames_to_account_id(self):
        sql = _relationship_table_sql("benefits_to", "cat", "sch")
        assert "transaction_id AS `source.transaction_id`" in sql
        assert "to_account_id AS `target.account_id`" in sql

    def test_has_position_columns(self):
        sql = _relationship_table_sql("has_position", "cat", "sch")
        assert "`source.account_id`" in sql
        assert "`target.position_id`" in sql
        assert "FROM `cat`.`sch`.`position`" in sql

    def test_of_security_columns(self):
        sql = _relationship_table_sql("of_security", "cat", "sch")
        assert "`source.position_id`" in sql
        assert "`target.stock_id`" in sql

    def test_creates_table(self):
        sql = _relationship_table_sql("has_account", "mycat", "mysch")
        assert sql.startswith("CREATE OR REPLACE TABLE `mycat`.`mysch`.`has_account`")


class TestSchemaConsistency:
    """Guard against drift between loading.RELATIONSHIP_DEFS and graph_schema.RELATIONSHIP_TABLES."""

    def test_relationship_defs_match_graph_schema_tables(self):
        loading_tables = set(RELATIONSHIP_DEFS.keys())
        schema_tables = {meta.delta_table for meta in RELATIONSHIP_TABLES.values()}
        assert loading_tables == schema_tables

    def test_relationship_alias_keys_match_graph_schema(self):
        # Build delta_table -> RelationshipMeta lookup
        by_table = {meta.delta_table: meta for meta in RELATIONSHIP_TABLES.values()}
        for rel_name, (_, _, src_alias, _, tgt_alias) in RELATIONSHIP_DEFS.items():
            meta = by_table[rel_name]
            assert src_alias == meta.source_key, (
                f"{rel_name}: source alias {src_alias!r} != schema key {meta.source_key!r}"
            )
            assert tgt_alias == meta.target_key, (
                f"{rel_name}: target alias {tgt_alias!r} != schema key {meta.target_key!r}"
            )
