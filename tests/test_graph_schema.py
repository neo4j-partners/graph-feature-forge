"""Tests for shared graph schema metadata."""

from graph_feature_forge.graph_schema import (
    BASE_NODE_LABELS,
    BASE_RELATIONSHIP_TYPES,
    NODE_CSV_FILES,
    NODE_TABLE_NAMES,
    NODE_TABLES,
    NodeMeta,
    RELATIONSHIP_TABLE_NAMES,
    RELATIONSHIP_TABLES,
    RelationshipMeta,
)


class TestNodeTables:
    def test_has_all_seven_entries(self):
        assert len(NODE_TABLES) == 7

    def test_expected_labels(self):
        expected = {"Customer", "Bank", "Account", "Company", "Stock", "Position", "Transaction"}
        assert set(NODE_TABLES.keys()) == expected

    def test_values_are_node_meta(self):
        for label, meta in NODE_TABLES.items():
            assert isinstance(meta, NodeMeta), f"{label} value should be NodeMeta"

    def test_named_field_access(self):
        meta = NODE_TABLES["Customer"]
        assert meta.key_property == "customer_id"
        assert meta.delta_table == "customer"

    def test_key_properties_end_with_id(self):
        for label, meta in NODE_TABLES.items():
            assert meta.key_property.endswith("_id"), f"{label} key should end with _id"

    def test_table_names_are_lowercase_labels(self):
        for label, meta in NODE_TABLES.items():
            assert meta.delta_table == label.lower(), f"{label} table should be {label.lower()}"

    def test_positional_unpacking_backward_compatible(self):
        key_prop, table = NODE_TABLES["Customer"]
        assert key_prop == "customer_id"
        assert table == "customer"


class TestRelationshipTables:
    def test_has_all_seven_entries(self):
        assert len(RELATIONSHIP_TABLES) == 7

    def test_expected_types(self):
        expected = {
            "HAS_ACCOUNT", "AT_BANK", "OF_COMPANY", "PERFORMS",
            "BENEFITS_TO", "HAS_POSITION", "OF_SECURITY",
        }
        assert set(RELATIONSHIP_TABLES.keys()) == expected

    def test_values_are_relationship_meta(self):
        for rel_type, meta in RELATIONSHIP_TABLES.items():
            assert isinstance(meta, RelationshipMeta), f"{rel_type} value should be RelationshipMeta"

    def test_named_field_access(self):
        meta = RELATIONSHIP_TABLES["HAS_ACCOUNT"]
        assert meta.source_label == "Customer"
        assert meta.source_key == "customer_id"
        assert meta.target_label == "Account"
        assert meta.target_key == "account_id"
        assert meta.delta_table == "has_account"

    def test_sources_and_targets_are_valid_nodes(self):
        for rel_type, meta in RELATIONSHIP_TABLES.items():
            assert meta.source_label in NODE_TABLES, f"{rel_type} source {meta.source_label} not a known node"
            assert meta.target_label in NODE_TABLES, f"{rel_type} target {meta.target_label} not a known node"

    def test_table_names_are_lowercase_types(self):
        for rel_type, meta in RELATIONSHIP_TABLES.items():
            assert meta.delta_table == rel_type.lower()

    def test_positional_unpacking_backward_compatible(self):
        src_label, src_key, tgt_label, tgt_key, table = RELATIONSHIP_TABLES["HAS_ACCOUNT"]
        assert src_label == "Customer"
        assert table == "has_account"


class TestNodeCsvFiles:
    def test_has_all_seven_csvs(self):
        assert len(NODE_CSV_FILES) == 7

    def test_position_maps_from_portfolio_holdings(self):
        assert NODE_CSV_FILES["portfolio_holdings.csv"] == "position"

    def test_all_tables_match_node_tables(self):
        csv_tables = set(NODE_CSV_FILES.values())
        node_tables = {table for _, (_, table) in NODE_TABLES.items()}
        assert csv_tables == node_tables


class TestBaseSets:
    def test_base_node_labels_match_node_tables(self):
        assert BASE_NODE_LABELS == frozenset(NODE_TABLES.keys())

    def test_base_rel_types_match_relationship_tables(self):
        assert BASE_RELATIONSHIP_TYPES == frozenset(RELATIONSHIP_TABLES.keys())

    def test_frozensets_are_immutable(self):
        assert isinstance(BASE_NODE_LABELS, frozenset)
        assert isinstance(BASE_RELATIONSHIP_TYPES, frozenset)


class TestDerivedLists:
    def test_node_table_names_count(self):
        assert len(NODE_TABLE_NAMES) == 7

    def test_relationship_table_names_count(self):
        assert len(RELATIONSHIP_TABLE_NAMES) == 7

    def test_node_table_names_are_strings(self):
        for name in NODE_TABLE_NAMES:
            assert isinstance(name, str)
            assert name == name.lower()

    def test_relationship_table_names_are_strings(self):
        for name in RELATIONSHIP_TABLE_NAMES:
            assert isinstance(name, str)
            assert name == name.lower()
