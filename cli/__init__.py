"""CLI wrapper — wires Runner to the semantic-auth project layout."""

from databricks_job_runner import Runner

runner = Runner(
    run_name_prefix="semantic_auth",
    wheel_package="semantic_auth",
)
