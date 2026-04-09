"""Configuration for the HTML document and embedding generator.

Reads settings from environment variables or .env file via Pydantic
Settings. All counts and paths have sensible defaults matching the
target scale defined in docs/grow.md.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class GeneratorConfig(BaseSettings):
    """Settings for HTML + embedding generation.

    Shared settings (``llm_endpoint``, ``embedding_endpoint``) read from
    the root env vars (``LLM_ENDPOINT``, ``EMBEDDING_ENDPOINT``) so a
    single ``.env`` configures the whole project.  Generator-specific
    settings keep the ``GEN_`` prefix.
    """

    model_config = {"env_prefix": "GEN_", "env_file": ".env", "extra": "ignore"}

    # --- Databricks endpoints (shared — no GEN_ prefix) ---
    # Reads LLM_ENDPOINT / EMBEDDING_ENDPOINT from root .env.
    # Falls back to GEN_LLM_ENDPOINT / GEN_EMBEDDING_ENDPOINT for
    # backwards compatibility.
    llm_endpoint: str = Field(
        default="databricks-claude-sonnet-4-6",
        validation_alias="LLM_ENDPOINT",
    )
    embedding_endpoint: str = Field(
        default="databricks-gte-large-en",
        validation_alias="EMBEDDING_ENDPOINT",
    )

    # --- Parallelism ---
    max_workers: int = 8

    # --- Document counts ---
    num_customer_profiles: int = 8
    num_company_analyses: int = 12
    num_sector_analyses: int = 7
    num_investment_guides: int = 7
    num_bank_profiles: int = 4
    num_regulatory_docs: int = 3

    # --- Chunking ---
    chunk_size: int = 4000
    chunk_overlap: int = 200

    # --- Embedding ---
    embedding_dimensions: int = 1024

    # --- Paths ---
    csv_dir: Path = Path("csv")
    html_output_dir: Path = Path("html")
    embedding_output_dir: Path = Path("embeddings")
    volume_base_path: str = "/Volumes/graph_feature_forge/enrichment/source-data"

    # --- Mode ---
    dry_run: bool = False
    seed: int = 42

    @property
    def total_documents(self) -> int:
        return (
            self.num_customer_profiles
            + self.num_company_analyses
            + self.num_sector_analyses
            + self.num_investment_guides
            + self.num_bank_profiles
            + self.num_regulatory_docs
        )
