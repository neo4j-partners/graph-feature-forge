"""Seed a Neo4j instance from Delta tables and embeddings JSON.

Replicates the graph-enrichment workshop's Lab 1 import flow:
1. Clear existing data
2. Create uniqueness constraints
3. Write 7 node types from Delta tables
4. Write 7 relationship types from Delta tables
5. Load document graph (Documents, Chunks, embeddings) from JSON
6. Create vector and full-text indexes

Uses the neo4j Python driver for all writes. Reads from Delta tables
via the existing SQLExecutor infrastructure (Spark or SDK).

The data volume is small (778 nodes, ~890 relationships) so batch
UNWIND writes via the driver are efficient.
"""

from __future__ import annotations

import json
from typing import Any

from semantic_auth.graph_schema import NODE_TABLES, RELATIONSHIP_TABLES
from semantic_auth.structured_data import SQLExecutor

#: Neo4j internal columns to drop when reading from Delta tables
_INTERNAL_COLS = {"<id>", "<labels>", "<rel.id>", "<rel.type>",
                  "<source.id>", "<source.labels>", "<target.id>", "<target.labels>"}


# ---------------------------------------------------------------------------
# Database preparation
# ---------------------------------------------------------------------------


def clear_database(session: Any) -> None:
    """Delete all nodes and relationships."""
    session.run("MATCH (n) DETACH DELETE n")
    print("    Cleared database")


def create_constraints(session: Any) -> None:
    """Create uniqueness constraints for all node types."""
    for label, (key_prop, _) in NODE_TABLES.items():
        session.run(
            f"CREATE CONSTRAINT {label.lower()}_{key_prop}_unique IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{key_prop} IS UNIQUE"
        )
    # Document graph constraints
    session.run(
        "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
        "FOR (n:Document) REQUIRE n.document_id IS UNIQUE"
    )
    session.run(
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
        "FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE"
    )
    print(f"    Created {len(NODE_TABLES) + 2} uniqueness constraints")


# ---------------------------------------------------------------------------
# Node writes
# ---------------------------------------------------------------------------


def _read_node_rows(
    execute_sql: SQLExecutor,
    catalog: str,
    schema: str,
    table: str,
) -> list[dict[str, Any]]:
    """Read rows from a node Delta table, dropping Neo4j internal columns."""
    rows = execute_sql(f"SELECT * FROM `{catalog}`.`{schema}`.`{table}`")
    cleaned = []
    for row in rows:
        cleaned.append({k: v for k, v in row.items() if k not in _INTERNAL_COLS})
    return cleaned


def write_nodes(
    session: Any,
    label: str,
    key_prop: str,
    rows: list[dict[str, Any]],
    batch_size: int = 100,
) -> int:
    """Write node rows to Neo4j via UNWIND batches."""
    if not rows:
        return 0

    # Build SET clause from all property keys
    prop_keys = [k for k in rows[0] if k != key_prop]
    set_clause = ", ".join(f"n.{k} = row.{k}" for k in prop_keys)

    query = (
        f"UNWIND $rows AS row "
        f"MERGE (n:{label} {{{key_prop}: row.{key_prop}}}) "
        f"SET {set_clause}"
    )

    written = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        session.run(query, {"rows": batch})
        written += len(batch)

    return written


# ---------------------------------------------------------------------------
# Relationship writes
# ---------------------------------------------------------------------------


def _read_relationship_keys(
    execute_sql: SQLExecutor,
    catalog: str,
    schema: str,
    table: str,
    source_key: str,
    target_key: str,
) -> list[dict[str, str]]:
    """Read source/target key pairs from a relationship Delta table."""
    rows = execute_sql(
        f"SELECT `source.{source_key}` AS source_key, "
        f"`target.{target_key}` AS target_key "
        f"FROM `{catalog}`.`{schema}`.`{table}`"
    )
    return rows


def write_relationships(
    session: Any,
    rel_type: str,
    source_label: str,
    source_key: str,
    target_label: str,
    target_key: str,
    rows: list[dict[str, str]],
    batch_size: int = 100,
) -> int:
    """Write relationship rows to Neo4j via UNWIND batches."""
    if not rows:
        return 0

    query = (
        f"UNWIND $rows AS row "
        f"MATCH (src:{source_label} {{{source_key}: row.source_key}}) "
        f"MATCH (tgt:{target_label} {{{target_key}: row.target_key}}) "
        f"MERGE (src)-[r:{rel_type}]->(tgt)"
    )

    written = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        session.run(query, {"rows": batch})
        written += len(batch)

    return written


# ---------------------------------------------------------------------------
# Document graph
# ---------------------------------------------------------------------------


def _read_embeddings_json(embeddings_path: str) -> dict:
    """Read embeddings JSON from a UC volume path."""
    with open(embeddings_path) as f:
        return json.load(f)


def load_document_graph(session: Any, embeddings_path: str) -> dict[str, int]:
    """Load Documents, Chunks, and embeddings from JSON file.

    Replicates the Lab 1 document graph import:
    - Document nodes with metadata
    - Chunk nodes with text and 1024-dim embeddings
    - FROM_DOCUMENT relationships (Chunk -> Document)
    - NEXT_CHUNK relationships (sequential chunks)
    - DESCRIBES relationships (Document -> Customer for profiles)
    """
    data = _read_embeddings_json(embeddings_path)

    documents = data["documents"]
    chunks = data["chunks"]
    counts: dict[str, int] = {}

    # Write Document nodes
    session.run(
        "UNWIND $documents AS doc "
        "MERGE (d:Document {document_id: doc.document_id}) "
        "SET d.filename = doc.filename, "
        "    d.document_type = doc.document_type, "
        "    d.title = doc.title, "
        "    d.source_path = doc.source_path, "
        "    d.char_count = doc.char_count",
        {"documents": documents},
    )
    counts["Document"] = len(documents)

    # Write Chunk nodes with embeddings (batch of 25 to stay under param limits)
    chunk_query = (
        "UNWIND $chunks AS chunk "
        "MERGE (c:Chunk {chunk_id: chunk.chunk_id}) "
        "SET c.text = chunk.text, "
        "    c.document_id = chunk.document_id, "
        "    c.`index` = chunk.index, "
        "    c.document_title = chunk.metadata.document_title, "
        "    c.document_type = chunk.metadata.document_type, "
        "    c.embedding = chunk.embedding"
    )
    for i in range(0, len(chunks), 25):
        session.run(chunk_query, {"chunks": chunks[i:i + 25]})
    counts["Chunk"] = len(chunks)

    # FROM_DOCUMENT relationships
    result = session.run(
        "MATCH (c:Chunk) WHERE c.document_id IS NOT NULL "
        "MATCH (d:Document {document_id: c.document_id}) "
        "MERGE (c)-[r:FROM_DOCUMENT]->(d) "
        "RETURN count(r) AS count"
    )
    counts["FROM_DOCUMENT"] = result.single()["count"]

    # NEXT_CHUNK relationships (sequential ordering)
    result = session.run(
        "MATCH (c1:Chunk) WHERE c1.document_id IS NOT NULL AND c1.index IS NOT NULL "
        "WITH c1 "
        "MATCH (c2:Chunk) "
        "WHERE c2.document_id = c1.document_id AND c2.index = c1.index + 1 "
        "MERGE (c1)-[r:NEXT_CHUNK]->(c2) "
        "RETURN count(r) AS count"
    )
    counts["NEXT_CHUNK"] = result.single()["count"]

    # DESCRIBES relationships (customer profile docs -> Customer nodes)
    result = session.run(
        "MATCH (d:Document) "
        "WHERE d.document_type = 'customer_profile' "
        "WITH d, replace(replace(d.title, 'Customer Profile - ', ''), "
        "                'Customer Profile: ', '') AS customer_name "
        "MATCH (c:Customer) "
        "WHERE c.first_name + ' ' + c.last_name = customer_name "
        "MERGE (d)-[r:DESCRIBES]->(c) "
        "RETURN count(r) AS count"
    )
    counts["DESCRIBES"] = result.single()["count"]

    return counts


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------


def create_indexes(driver: Any, database: str, wait: bool = True, timeout: int = 120) -> None:
    """Create vector and full-text indexes on Chunk nodes.

    Uses separate sessions for each DDL command (Neo4j Aura requires
    schema commands to run in their own auto-commit transactions on
    fresh sessions).

    Args:
        driver: Neo4j driver instance.
        database: Neo4j database name.
        wait: If True, poll until indexes are ONLINE before returning.
        timeout: Maximum seconds to wait for indexes to come online.
    """
    import time

    # Log server info for debugging
    with driver.session(database=database) as s:
        info = s.run("CALL dbms.components() YIELD name, versions RETURN name, versions").single()
        if info:
            print(f"    Neo4j: {info['name']} {info['versions']}")

        # Show existing indexes before creation
        existing = list(s.run("SHOW INDEXES YIELD name, type, state RETURN name, type, state"))
        if existing:
            print(f"    Existing indexes: {[(r['name'], r['type'], r['state']) for r in existing]}")
        else:
            print("    No existing indexes")

    # Create each index in its own session
    index_cmds = [
        (
            "chunk_embedding_index",
            "CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.embedding) "
            "OPTIONS {indexConfig: {"
            "  `vector.dimensions`: 1024, "
            "  `vector.similarity_function`: 'cosine'"
            "}}",
        ),
        (
            "chunk_text_index",
            "CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS "
            "FOR (c:Chunk) ON EACH [c.text]",
        ),
    ]

    for idx_name, cypher in index_cmds:
        try:
            with driver.session(database=database) as s:
                s.run(cypher).consume()
            print(f"    {idx_name}: CREATE command OK")
        except Exception as exc:
            print(f"    {idx_name}: CREATE failed: {type(exc).__name__}: {exc}")

    # Verify indexes exist after creation
    with driver.session(database=database) as s:
        after = list(s.run("SHOW INDEXES YIELD name, type, state RETURN name, type, state"))
        print(f"    Indexes after creation: {[(r['name'], r['type'], r['state']) for r in after]}")

    if not wait:
        return

    target_indexes = {"chunk_embedding_index", "chunk_text_index"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        with driver.session(database=database) as s:
            result = s.run(
                "SHOW INDEXES YIELD name, state "
                "WHERE name IN $names "
                "RETURN name, state",
                names=list(target_indexes),
            )
            statuses = {record["name"]: record["state"] for record in result}
        if all(statuses.get(n) == "ONLINE" for n in target_indexes):
            print("    Indexes ONLINE")
            return
        time.sleep(2)

    pending = {n: statuses.get(n, "MISSING") for n in target_indexes
               if statuses.get(n) != "ONLINE"}
    print(f"    WARNING: indexes not ONLINE after {timeout}s: {pending}")


# ---------------------------------------------------------------------------
# Main seeding orchestrator
# ---------------------------------------------------------------------------


def seed_neo4j(
    execute_sql: SQLExecutor,
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    neo4j_database: str,
    source_catalog: str,
    source_schema: str,
    embeddings_path: str,
) -> dict[str, int]:
    """Seed a Neo4j instance from Delta tables and embeddings JSON.

    This is the main entry point. It:
    1. Clears the database
    2. Creates constraints
    3. Writes all node types
    4. Writes all relationship types
    5. Loads the document graph
    6. Creates indexes

    Returns a dict mapping entity names to row counts.
    """
    import neo4j as neo4j_driver

    driver = neo4j_driver.GraphDatabase.driver(
        neo4j_uri, auth=(neo4j_username, neo4j_password),
    )
    counts: dict[str, int] = {}

    try:
        with driver.session(database=neo4j_database) as session:
            # Step 1: Clear and prepare
            print("\n  Preparing database ...")
            clear_database(session)
            create_constraints(session)

            # Step 2: Write nodes
            print("\n  Writing nodes ...")
            for label, (key_prop, table) in NODE_TABLES.items():
                rows = _read_node_rows(execute_sql, source_catalog, source_schema, table)
                n = write_nodes(session, label, key_prop, rows)
                counts[label] = n
                print(f"    {label}: {n} nodes")

            # Step 3: Write relationships
            print("\n  Writing relationships ...")
            for rel_type, (src_label, src_key, tgt_label, tgt_key, table) in RELATIONSHIP_TABLES.items():
                rows = _read_relationship_keys(
                    execute_sql, source_catalog, source_schema, table, src_key, tgt_key,
                )
                n = write_relationships(
                    session, rel_type, src_label, src_key, tgt_label, tgt_key, rows,
                )
                counts[rel_type] = n
                print(f"    {rel_type}: {n} relationships")

            # Step 4: Document graph
            print("\n  Loading document graph ...")
            doc_counts = load_document_graph(session, embeddings_path)
            counts.update(doc_counts)
            for name, n in doc_counts.items():
                print(f"    {name}: {n}")

        # Step 5: Indexes (uses separate sessions per DDL command)
        print("\n  Creating indexes ...")
        create_indexes(driver, neo4j_database)

    finally:
        driver.close()

    total_nodes = sum(counts.get(label, 0) for label in NODE_TABLES) + counts.get("Document", 0) + counts.get("Chunk", 0)
    total_rels = sum(counts.get(rt, 0) for rt in RELATIONSHIP_TABLES) + counts.get("FROM_DOCUMENT", 0) + counts.get("NEXT_CHUNK", 0) + counts.get("DESCRIBES", 0)
    print(f"\n  Seeding complete: {total_nodes} nodes, {total_rels} relationships")

    return counts
