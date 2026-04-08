"""Quick test: does the Neo4j Spark Connector work on this cluster?"""
from __future__ import annotations
import os
import sys
import signal

def timeout_handler(signum, frame):
    print("TIMEOUT: Spark Connector read hung after 60 seconds")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)

def main():
    from graph_feature_forge import inject_params
    inject_params()

    from databricks.sdk import WorkspaceClient
    wc = WorkspaceClient()
    print(f"Connected to {wc.config.host}")

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_pass = os.getenv("NEO4J_PASSWORD")
    neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")

    print(f"Neo4j URI: {neo4j_uri}")

    # Test 1: Python driver (should work)
    print("\n--- Test 1: Neo4j Python driver ---")
    import neo4j
    driver = neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    with driver.session(database=neo4j_db) as session:
        count = session.run("MATCH (c:Customer) RETURN count(c) AS cnt").single()["cnt"]
        print(f"  Customer count via Python driver: {count}")
    driver.close()

    # Test 2: Spark Connector
    print("\n--- Test 2: Neo4j Spark Connector ---")
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

    print("  Setting 60s alarm...")
    signal.alarm(60)

    print("  Calling spark.read.format('org.neo4j.spark.DataSource').load()...")
    try:
        df = (
            spark.read.format("org.neo4j.spark.DataSource")
            .option("url", neo4j_uri)
            .option("authentication.type", "basic")
            .option("authentication.basic.username", neo4j_user)
            .option("authentication.basic.password", neo4j_pass)
            .option("database", neo4j_db)
            .option("labels", ":Customer")
            .load()
        )
        signal.alarm(0)
        print(f"  Schema loaded: {df.columns[:5]}...")
        print(f"  Row count: {df.count()}")
        print("  SUCCESS")
    except Exception as e:
        signal.alarm(0)
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")

if __name__ == "__main__":
    main()
