"""Tests for config Neo4j fields."""

from graph_feature_forge.config import Config


class TestConfigNeo4j:
    def test_defaults(self):
        c = Config()
        assert c.neo4j_uri is None
        assert c.neo4j_username is None
        assert c.neo4j_password is None
        assert c.neo4j_database == "neo4j"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("NEO4J_URI", "neo4j+s://test.databases.neo4j.io")
        monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
        monkeypatch.setenv("NEO4J_PASSWORD", "secret")
        monkeypatch.setenv("NEO4J_DATABASE", "mydb")

        c = Config.from_env()
        assert c.neo4j_uri == "neo4j+s://test.databases.neo4j.io"
        assert c.neo4j_username == "neo4j"
        assert c.neo4j_password == "secret"
        assert c.neo4j_database == "mydb"

    def test_database_default_when_not_set(self, monkeypatch):
        # Ensure NEO4J_DATABASE is not set
        monkeypatch.delenv("NEO4J_DATABASE", raising=False)
        c = Config.from_env()
        assert c.neo4j_database == "neo4j"
