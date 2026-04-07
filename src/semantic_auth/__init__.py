"""semantic-auth: Graph enrichment without Genie, Knowledge Agents, or Supervisor Agent."""

from __future__ import annotations

import os
import sys

__version__ = "0.1.0"


def inject_params() -> None:
    """Parse ``KEY=VALUE`` parameters from ``sys.argv`` into ``os.environ``.

    Re-exports :func:`databricks_job_runner.inject_params` when available
    (local dev), otherwise provides an identical inline implementation
    for the serverless runtime where only the wheel is installed.
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
