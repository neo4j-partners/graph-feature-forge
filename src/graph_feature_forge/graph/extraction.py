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

from collections.abc import Set
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


def spark_neo4j_options(
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
    overwrite_schema: bool = False,
) -> int:
    """Extract all nodes with the given label to a Delta table.

    Returns the number of rows written.
    """
    options = spark_neo4j_options(uri, username, password, database)
    options["labels"] = f":{label}"

    df = (
        spark.read.format("org.neo4j.spark.DataSource")
        .options(**options)
        .load()
    )

    table_name = f"`{catalog}`.`{schema}`.`{label.lower()}`"
    writer = df.write.format("delta").mode("overwrite")
    if overwrite_schema:
        writer = writer.option("overwriteSchema", "true")
    writer.saveAsTable(table_name)

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
    overwrite_schema: bool = False,
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
    options = spark_neo4j_options(uri, username, password, database)
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
    writer = df.write.format("delta").mode("overwrite")
    if overwrite_schema:
        writer = writer.option("overwriteSchema", "true")
    writer.saveAsTable(table_name)

    count = df.count()
    print(f"    {rel_type} -> {table_name}: {count} rows")
    return count


# NOTE: enrichment.md proposes timestamp-based incremental extraction
# (filter by enrichment_timestamp after the Spark Connector reads).
# This is deferred: volumes are small (hundreds of rows per type),
# and incrementality is handled at the enrichment_log level via
# deduplication and prior-enrichment context in synthesis prompts.


def extract_graph(
    spark: Any,
    uri: str,
    username: str,
    password: str,
    database: str,
    catalog: str,
    schema: str,
    *,
    base_node_labels: Set[str] = frozenset(),
    base_rel_types: Set[str] = frozenset(),
) -> dict[str, int]:
    """Extract graph data from Neo4j to Delta tables.

    Dynamically discovers all node labels and relationship types,
    then extracts each one.  When *base_node_labels* or
    *base_rel_types* are provided, those labels/types are skipped —
    their Delta tables were created by ``loading.py`` with explicit
    type CASTs and should not be overwritten by Spark Connector–
    inferred types.

    Enrichment labels/types (those **not** in the base sets) are
    extracted with ``overwriteSchema=true`` so their Delta tables
    can evolve across runs as new properties appear.

    Returns a dict mapping table names to row counts.
    """
    print("\n  Discovering Neo4j schema ...")
    labels, rel_types, rel_endpoints = discover_schema(
        uri, username, password, database,
    )
    print(f"    Node labels: {labels}")
    print(f"    Relationship types: {rel_types}")

    enrichment_labels = [lbl for lbl in labels if lbl not in base_node_labels]
    enrichment_rels = [r for r in rel_types if r not in base_rel_types]

    if base_node_labels or base_rel_types:
        skipped_nodes = len(labels) - len(enrichment_labels)
        skipped_rels = len(rel_types) - len(enrichment_rels)
        print(
            f"    Selective mode: skipping {skipped_nodes} base node labels, "
            f"{skipped_rels} base relationship types"
        )

    counts: dict[str, int] = {}

    if enrichment_labels:
        print("\n  Extracting enrichment nodes ...")
        for label in enrichment_labels:
            try:
                n = extract_nodes(
                    spark, label, uri, username, password, database,
                    catalog, schema, overwrite_schema=True,
                )
                counts[label.lower()] = n
            except Exception as exc:
                print(f"    {label}: FAILED — {exc}")

    if enrichment_rels:
        print("\n  Extracting enrichment relationships ...")
        for rel_type in enrichment_rels:
            src_label, tgt_label = rel_endpoints.get(rel_type, ("", ""))
            try:
                n = extract_relationships(
                    spark, rel_type, src_label, tgt_label,
                    uri, username, password, database,
                    catalog, schema, overwrite_schema=True,
                )
                counts[rel_type.lower()] = n
            except Exception as exc:
                print(f"    {rel_type}: FAILED — {exc}")

    total = sum(counts.values())
    if counts:
        print(f"\n  Extraction complete: {len(counts)} tables, {total} total rows")
    else:
        print("\n  No enrichment data to extract")
    return counts
