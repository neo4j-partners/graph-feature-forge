"""graph_feature_forge: Graph enrichment without Genie, Knowledge Agents, or Supervisor Agent."""

from __future__ import annotations

import os
import sys

__version__ = "0.1.0"


def inject_params() -> None:
    """Parse ``KEY=VALUE`` parameters from ``sys.argv`` into ``os.environ``.

    Re-exports :func:`databricks_job_runner.inject_params` when available
    (local dev), otherwise provides an identical inline implementation
    for the serverless runtime where only the wheel is installed.

    After loading parameters, fetches any keys listed in
    ``DATABRICKS_SECRET_KEYS`` from the Databricks secret scope.
    """
    try:
        from databricks_job_runner import inject_params as _inject

        _inject()
    except ImportError:
        remaining: list[str] = []
        for arg in sys.argv[1:]:
            if "=" in arg and not arg.startswith("-"):
                key, _, value = arg.partition("=")
                os.environ.setdefault(key, value)
            else:
                remaining.append(arg)
        sys.argv[1:] = remaining
        _load_secrets()


def _load_secrets() -> None:
    """Fetch secrets from a Databricks secret scope into ``os.environ``.

    Mirrors :func:`databricks_job_runner.inject._load_secrets` for the
    serverless runtime where ``databricks-job-runner`` is not installed.
    """
    scope = os.environ.get("DATABRICKS_SECRET_SCOPE")
    raw_keys = os.environ.get("DATABRICKS_SECRET_KEYS")
    if not scope or not raw_keys:
        return
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    if not keys:
        return

    from databricks.sdk import WorkspaceClient

    ws = WorkspaceClient()
    for key in keys:
        try:
            value = ws.dbutils.secrets.get(scope=scope, key=key)
            os.environ.setdefault(key, value)
        except Exception as exc:
            print(
                f"WARNING: failed to load secret '{key}' "
                f"from scope '{scope}': {exc}"
            )
