"""Neo4j graph extraction to Delta tables via Spark Connector.

Dynamically discovers all node labels and relationship types in the
Neo4j database and extracts each as a Delta table.  Uses the neo4j
Python driver for schema discovery and the Neo4j Spark Connector for
bulk data extraction.

The output format matches the Lab 4 workshop export:
- Node tables: one per label, columns are node properties
- Relationship tables: one per type, source/target node properties
  prefixed with ``source.`` and ``target.`` (connector convention
  when ``relationship.nodes.map=false``)
"""

from __future__ import annotations

from typing import Any


def discover_schema(
    uri: str,
    username: str,
    password: str,
    database: str = "neo4j",
) -> tuple[list[str], list[str], dict[str, tuple[str, str]]]:
    """Query Neo4j for all node labels, relationship types, and endpoints.

    Returns:
        Tuple of (node_labels, relationship_types, rel_endpoints).
        rel_endpoints maps each relationship type to its
        (source_label, target_label) pair.
    """
    import neo4j

    driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            labels = session.run("CALL db.labels()").value()
            rel_types = session.run("CALL db.relationshipTypes()").value()

            # Discover source/target labels per relationship type
            result = session.run(
                "MATCH (s)-[r]->(t) "
                "WITH type(r) AS rel_type, labels(s)[0] AS src, labels(t)[0] AS tgt "
                "RETURN DISTINCT rel_type, src, tgt"
            )
            rel_endpoints: dict[str, tuple[str, str]] = {}
            for record in result:
                rel_endpoints[record["rel_type"]] = (
                    record["src"],
                    record["tgt"],
                )
    finally:
        driver.close()

    return sorted(labels), sorted(rel_types), rel_endpoints


def _spark_neo4j_options(
    uri: str,
    username: str,
    password: str,
    database: str,
) -> dict[str, str]:
    """Base Spark Connector options shared by node and relationship reads."""
    return {
        "url": uri,
        "authentication.type": "basic",
        "authentication.basic.username": username,
        "authentication.basic.password": password,
        "database": database,
    }


def extract_nodes(
    spark: Any,
    label: str,
    uri: str,
    username: str,
    password: str,
    database: str,
    catalog: str,
    schema: str,
) -> int:
    """Extract all nodes with the given label to a Delta table.

    Returns the number of rows written.
    """
    options = _spark_neo4j_options(uri, username, password, database)
    options["labels"] = f":{label}"

    df = (
        spark.read.format("org.neo4j.spark.DataSource")
        .options(**options)
        .load()
    )

    table_name = f"`{catalog}`.`{schema}`.`{label.lower()}`"
    df.write.format("delta").mode("overwrite").saveAsTable(table_name)

    count = df.count()
    print(f"    {label} -> {table_name}: {count} rows")
    return count


def extract_relationships(
    spark: Any,
    rel_type: str,
    source_label: str,
    target_label: str,
    uri: str,
    username: str,
    password: str,
    database: str,
    catalog: str,
    schema: str,
) -> int:
    """Extract all relationships of the given type to a Delta table.

    Uses ``relationship.nodes.map=false`` to match the Lab 4 export
    convention where source/target properties are prefixed with
    ``source.`` and ``target.``.

    Args:
        source_label: Node label on the source end of the relationship.
        target_label: Node label on the target end of the relationship.

    Returns the number of rows written.
    """
    options = _spark_neo4j_options(uri, username, password, database)
    options["relationship"] = rel_type
    options["relationship.nodes.map"] = "false"
    options["relationship.source.labels"] = f":{source_label}"
    options["relationship.target.labels"] = f":{target_label}"

    df = (
        spark.read.format("org.neo4j.spark.DataSource")
        .options(**options)
        .load()
    )

    table_name = f"`{catalog}`.`{schema}`.`{rel_type.lower()}`"
    df.write.format("delta").mode("overwrite").saveAsTable(table_name)

    count = df.count()
    print(f"    {rel_type} -> {table_name}: {count} rows")
    return count


def extract_graph(
    spark: Any,
    uri: str,
    username: str,
    password: str,
    database: str,
    catalog: str,
    schema: str,
) -> dict[str, int]:
    """Extract the full graph from Neo4j to Delta tables.

    Dynamically discovers all node labels and relationship types,
    then extracts each one.  Tables are overwritten so the Delta
    tables always reflect the current graph state.

    Returns a dict mapping table names to row counts.
    """
    print("\n  Discovering Neo4j schema ...")
    labels, rel_types, rel_endpoints = discover_schema(
        uri, username, password, database,
    )
    print(f"    Node labels: {labels}")
    print(f"    Relationship types: {rel_types}")

    counts: dict[str, int] = {}

    print("\n  Extracting nodes ...")
    for label in labels:
        try:
            n = extract_nodes(
                spark, label, uri, username, password, database, catalog, schema,
            )
            counts[label.lower()] = n
        except Exception as exc:
            print(f"    {label}: FAILED — {exc}")

    print("\n  Extracting relationships ...")
    for rel_type in rel_types:
        src_label, tgt_label = rel_endpoints.get(rel_type, ("", ""))
        try:
            n = extract_relationships(
                spark, rel_type, src_label, tgt_label,
                uri, username, password, database, catalog, schema,
            )
            counts[rel_type.lower()] = n
        except Exception as exc:
            print(f"    {rel_type}: FAILED — {exc}")

    total = sum(counts.values())
    print(f"\n  Extraction complete: {len(counts)} tables, {total} total rows")
    return counts
