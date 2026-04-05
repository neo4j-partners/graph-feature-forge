"""Configuration for the semantic-auth enrichment pipeline.

Loads settings from environment variables with sensible defaults.
On Databricks clusters, WorkspaceClient auto-discovers credentials.
Locally, set DATABRICKS_HOST and DATABRICKS_TOKEN (or use a
~/.databrickscfg profile).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Pipeline configuration."""

    # Workshop Delta tables (source data from Lab 4 export)
    source_catalog: str = "neo4j_augmentation_demo"
    source_schema: str = "raw_data"

    # semantic-auth catalog (enrichment artifacts)
    catalog: str = "semantic-graph-enrichment"
    schema: str = "enrichment"
    volume: str = "source-data"

    # Model endpoints
    llm_endpoint: str = "databricks-claude-sonnet-4-6"
    embedding_endpoint: str = "databricks-gte-large-en"

    # SQL warehouse ID (required for local SDK-based SQL execution)
    warehouse_id: str | None = None

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
        )
