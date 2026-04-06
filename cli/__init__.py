"""CLI wrapper — wires Runner to the semantic-auth project layout."""

from databricks_job_runner import Runner, RunnerConfig

#: Keys from .env that get forwarded as CLI flags to submitted scripts.
_PARAM_KEYS = {
    "SOURCE_CATALOG": "--source-catalog",
    "SOURCE_SCHEMA": "--source-schema",
    "CATALOG_NAME": "--catalog-name",
    "SCHEMA_NAME": "--schema-name",
    "VOLUME_NAME": "--volume-name",
    "LLM_ENDPOINT": "--llm-endpoint",
    "EMBEDDING_ENDPOINT": "--embedding-endpoint",
    "WAREHOUSE_ID": "--warehouse-id",
}


def build_params(config: RunnerConfig, script: str = "") -> list[str]:
    """Forward semantic-auth config keys from ``.env`` as CLI flags."""
    params: list[str] = []
    for key, flag in _PARAM_KEYS.items():
        value = config.extras.get(key)
        if value:
            params += [flag, value]
    return params


runner = Runner(
    run_name_prefix="semantic_auth",
    build_params=build_params,
    wheel_package="semantic_auth",
)
