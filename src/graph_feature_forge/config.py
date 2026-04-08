"""Configuration for the graph-feature-forge pipeline.

Loads settings from environment variables with sensible defaults.
On Databricks clusters, WorkspaceClient auto-discovers credentials.
Locally, set DATABRICKS_HOST and DATABRICKS_TOKEN (or use a
~/.databrickscfg profile).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Pipeline configuration."""

    # Workshop Delta tables (source data from Lab 4 export)
    source_catalog: str = "neo4j_augmentation_demo"
    source_schema: str = "raw_data"

    # graph-feature-forge catalog (enrichment artifacts)
    catalog: str = "graph-feature-forge"
    schema: str = "enrichment"
    volume: str = "source-data"

    # Model endpoints
    llm_endpoint: str = "databricks-claude-sonnet-4-6"
    embedding_endpoint: str = "databricks-gte-large-en"

    # SQL warehouse ID (required for local SDK-based SQL execution)
    warehouse_id: str | None = None

    # Neo4j connection (required for extraction and write-back)
    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"

    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables."""
        return cls(
            source_catalog=os.getenv("SOURCE_CATALOG", cls.source_catalog),
            source_schema=os.getenv("SOURCE_SCHEMA", cls.source_schema),
            catalog=os.getenv("CATALOG_NAME", cls.catalog),
            schema=os.getenv("SCHEMA_NAME", cls.schema),
            volume=os.getenv("VOLUME_NAME", cls.volume),
            llm_endpoint=os.getenv("LLM_ENDPOINT", cls.llm_endpoint),
            embedding_endpoint=os.getenv("EMBEDDING_ENDPOINT", cls.embedding_endpoint),
            warehouse_id=os.getenv("WAREHOUSE_ID"),
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_username=os.getenv("NEO4J_USERNAME"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            neo4j_database=os.getenv("NEO4J_DATABASE", cls.neo4j_database),
        )
