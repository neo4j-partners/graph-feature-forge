"""Tests for loading module — verifies SQL generation."""

from semantic_auth.loading import (
    NODE_CSV_TABLES,
    RELATIONSHIP_DEFS,
    _NODE_SQL,
    _node_table_sql,
    _relationship_table_sql,
)


class TestNodeCsvTables:
    def test_all_seven_csvs_mapped(self):
        assert len(NODE_CSV_TABLES) == 7

    def test_position_maps_from_portfolio_holdings(self):
        assert NODE_CSV_TABLES["portfolio_holdings.csv"] == "position"

    def test_all_sql_templates_exist(self):
        for table_name in NODE_CSV_TABLES.values():
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
