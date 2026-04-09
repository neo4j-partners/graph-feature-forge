"""CLI wrapper — wires Runner to the graph_feature_forge project layout."""

from databricks_job_runner import Runner

# Keys whose values live in a Databricks secret scope rather than
# being passed as plaintext job parameters.
SECRET_KEYS = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]

runner = Runner(
    run_name_prefix="graph_feature_forge",
    wheel_package="graph_feature_forge",
    secret_keys=SECRET_KEYS,
)
