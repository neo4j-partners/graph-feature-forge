"""CLI wrapper — wires Runner to the semantic-auth project layout."""

from databricks_job_runner import Runner

# Keys whose values live in a Databricks secret scope rather than
# being passed as plaintext job parameters.
SECRET_KEYS = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]

runner = Runner(
    run_name_prefix="semantic_auth",
    wheel_package="semantic_auth",
    secret_keys=SECRET_KEYS,
)
