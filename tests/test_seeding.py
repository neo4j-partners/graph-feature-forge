"""Tests for seeding module helpers."""

from graph_feature_forge.seeding import (
    NODE_TABLES,
    RELATIONSHIP_TABLES,
    _INTERNAL_COLS,
    _read_node_rows,
)


class TestSchemaDefinitions:
    def test_all_node_tables_defined(self):
        expected = {"Customer", "Bank", "Account", "Company", "Stock", "Position", "Transaction"}
        assert set(NODE_TABLES.keys()) == expected

    def test_all_relationship_tables_defined(self):
        expected = {
            "HAS_ACCOUNT", "AT_BANK", "OF_COMPANY", "PERFORMS",
            "BENEFITS_TO", "HAS_POSITION", "OF_SECURITY",
        }
        assert set(RELATIONSHIP_TABLES.keys()) == expected

    def test_node_table_tuples(self):
        for label, (key_prop, table) in NODE_TABLES.items():
            assert key_prop.endswith("_id"), f"{label} key should end with _id"
            assert table == label.lower(), f"{label} table should be lowercase label"

    def test_relationship_table_tuples(self):
        for rel_type, (src_label, src_key, tgt_label, tgt_key, table) in RELATIONSHIP_TABLES.items():
            assert src_label in NODE_TABLES, f"{rel_type} source {src_label} not a known node"
            assert tgt_label in NODE_TABLES, f"{rel_type} target {tgt_label} not a known node"
            assert table == rel_type.lower(), f"{rel_type} table should be lowercase"


class TestReadNodeRows:
    def test_strips_internal_columns(self):
        fake_rows = [
            {"<id>": 1, "<labels>": ["Customer"], "customer_id": "C0001", "name": "Test"},
            {"<id>": 2, "<labels>": ["Customer"], "customer_id": "C0002", "name": "Test2"},
        ]

        def fake_executor(query):
            return fake_rows

        result = _read_node_rows(fake_executor, "cat", "sch", "customer")
        assert len(result) == 2
        assert "<id>" not in result[0]
        assert "<labels>" not in result[0]
        assert result[0]["customer_id"] == "C0001"
        assert result[0]["name"] == "Test"


class TestInternalCols:
    def test_covers_node_internals(self):
        assert "<id>" in _INTERNAL_COLS
        assert "<labels>" in _INTERNAL_COLS

    def test_covers_relationship_internals(self):
        assert "<rel.id>" in _INTERNAL_COLS
        assert "<rel.type>" in _INTERNAL_COLS
        assert "<source.id>" in _INTERNAL_COLS
        assert "<target.id>" in _INTERNAL_COLS
