"""Tests for extraction helper functions."""

from graph_feature_forge.graph.extraction import spark_neo4j_options


class TestSparkNeo4jOptions:
    def test_basic_options(self):
        opts = spark_neo4j_options(
            uri="neo4j+s://test.io",
            username="neo4j",
            password="secret",
            database="mydb",
        )
        assert opts["url"] == "neo4j+s://test.io"
        assert opts["authentication.type"] == "basic"
        assert opts["authentication.basic.username"] == "neo4j"
        assert opts["authentication.basic.password"] == "secret"
        assert opts["database"] == "mydb"

    def test_all_keys_present(self):
        opts = spark_neo4j_options("uri", "user", "pass", "db")
        expected_keys = {
            "url", "authentication.type",
            "authentication.basic.username",
            "authentication.basic.password",
            "database",
        }
        assert set(opts.keys()) == expected_keys
