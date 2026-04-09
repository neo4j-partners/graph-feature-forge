"""GDS Demo: PageRank, Louvain, Node Similarity + Fraud Detection Report.

Demonstrates the core GDS workflow — project the portfolio graph, run three
algorithms, and write computed properties back to Account nodes in Neo4j.

    +-----------+------------------+----------------------------------------------+
    | Algorithm | Property Written | Business Meaning                             |
    +-----------+------------------+----------------------------------------------+
    | PageRank  | risk_score       | How central is this account to suspicious     |
    |           |                  | activity?                                    |
    | Louvain   | community_id     | Which cluster of accounts behave similarly?  |
    | Node      | similarity_score | Which accounts share the most counterparties?|
    | Similarity|                  |                                              |
    +-----------+------------------+----------------------------------------------+

Usage:
    python -m cli upload --wheel
    python -m cli upload gds_demo/gds_demo.py
    python -m cli submit gds_demo.py --compute cluster
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants — define the projection and algorithm targets
# ---------------------------------------------------------------------------

PROJECTION_NAME = "gds-demo"

NODE_LABELS = ["Customer", "Account", "Bank", "Position", "Stock", "Company", "Transaction"]

RELATIONSHIP_SPEC = {
    "HAS_ACCOUNT": {"orientation": "UNDIRECTED"},
    "AT_BANK": {"orientation": "UNDIRECTED"},
    "HAS_POSITION": {"orientation": "UNDIRECTED"},
    "OF_SECURITY": {"orientation": "UNDIRECTED"},
    "OF_COMPANY": {"orientation": "UNDIRECTED"},
    "PERFORMS": {"orientation": "UNDIRECTED"},
    "BENEFITS_TO": {"orientation": "UNDIRECTED"},
}

REPORT_PATH = os.getenv("FRAUD_REPORT_PATH", "fraud_report.json")


# ---------------------------------------------------------------------------
# Step functions — each one does exactly one thing
# ---------------------------------------------------------------------------


def connect(uri: str, username: str, password: str, database: str):
    """Create a GDS client connected to the Neo4j instance."""
    from graphdatascience import GraphDataScience

    gds = GraphDataScience(uri, auth=(username, password), database=database)
    print(f"  Connected to GDS {gds.version()}")
    return gds


def create_projection(gds):
    """Project the portfolio graph into GDS memory.

    Includes Customer, Account, Bank, Position, Stock, Company, and
    Transaction nodes with seven undirected relationship types.
    Transaction connectivity enables GDS algorithms to detect money
    flow patterns such as circular transfers and structuring.

    Returns the projected graph object G.
    """
    # Drop any leftover projection from a previous run
    try:
        gds.graph.drop(gds.graph.get(PROJECTION_NAME))
    except Exception:
        pass

    G, _ = gds.graph.project(PROJECTION_NAME, NODE_LABELS, RELATIONSHIP_SPEC)
    print(
        f"  Projected '{PROJECTION_NAME}' — {G.node_count()} nodes, {G.relationship_count()} relationships"
    )
    return G


def run_pagerank(gds, G):
    """Run PageRank and mutate risk_score onto every node in the projection.

    Higher risk_score means the account sits at a hub in the financial
    network — it connects to many other accounts, positions, and companies.
    """
    result = gds.pageRank.mutate(
        G,
        mutateProperty="risk_score",
        maxIterations=20,
        dampingFactor=0.85,
    )
    print(
        f"  PageRank: {result['nodePropertiesWritten']} nodes scored, "
        f"ran {result['ranIterations']} iterations, converged={result['didConverge']}"
    )
    return result


def run_louvain(gds, G):
    """Run Louvain community detection and mutate community_id.

    Nodes that are more densely connected to each other than to the rest
    of the graph end up in the same community.  In a portfolio graph this
    surfaces clusters of accounts that share banks, stocks, or counterparties.
    """
    result = gds.louvain.mutate(
        G,
        mutateProperty="community_id",
        maxLevels=10,
        maxIterations=10,
    )
    print(
        f"  Louvain: {result['communityCount']} communities, "
        f"modularity {result['modularity']:.4f}"
    )
    return result


def run_node_similarity(gds, G):
    """Run Node Similarity and mutate SIMILAR_TO relationships.

    Computes pairwise Jaccard similarity between nodes based on shared
    neighbors.  Accounts that hold positions in the same stocks or share
    the same bank will score highly.

    Node Similarity produces *relationships*, not a node property.  The
    write_properties step below aggregates these into a per-account
    similarity_score.
    """
    result = gds.nodeSimilarity.mutate(
        G,
        mutateRelationshipType="SIMILAR_TO",
        mutateProperty="similarity_score",
        similarityCutoff=0.1,
        topK=10,
    )
    dist = result["similarityDistribution"]
    print(
        f"  Node Similarity: {result['nodesCompared']} nodes compared, "
        f"{result['relationshipsWritten']} relationships created"
    )
    print(
        f"  Distribution — min: {dist['min']:.4f}, mean: {dist['mean']:.4f}, max: {dist['max']:.4f}"
    )
    return result


def write_properties(gds, G):
    """Write computed properties back to Neo4j Account nodes.

    1. Write risk_score and community_id as node properties via GDS.
    2. Write SIMILAR_TO relationships to Neo4j.
    3. Aggregate max similarity per account into a similarity_score node property.
    """
    # --- Node properties (PageRank + Louvain) ---
    gds.graph.nodeProperties.write(
        G,
        node_properties=["risk_score", "community_id"],
        node_labels=["Account"],
    )
    print("  Written risk_score and community_id to Account nodes")

    # --- Similarity relationships ---
    gds.graph.relationship.write(
        G,
        relationship_type="SIMILAR_TO",
        relationship_property="similarity_score",
    )
    print("  Written SIMILAR_TO relationships to Neo4j")

    # --- Aggregate similarity into a node property ---
    # GDS mutate produces directed SIMILAR_TO rels even on undirected graphs.
    # Undirected match -[r]-() captures both incoming and outgoing so every
    # participating Account gets a score.
    agg_result = gds.run_cypher("""
        MATCH (a:Account)-[r:SIMILAR_TO]-()
        WITH a, max(r.similarity_score) AS max_sim
        SET a.similarity_score = max_sim
        RETURN count(a) AS updated
    """)
    print(
        f"  Aggregated max similarity_score onto {agg_result.iloc[0]['updated']} Account nodes"
    )


def verify_results(gds):
    """Run verification queries to confirm properties were written."""
    result = gds.run_cypher("""
        MATCH (a:Account)
        RETURN
            count(a) AS total,
            count(a.risk_score) AS has_risk,
            count(a.community_id) AS has_community,
            count(a.similarity_score) AS has_similarity,
            avg(a.risk_score) AS avg_risk,
            size(collect(DISTINCT a.community_id)) AS num_communities
    """)
    row = result.iloc[0]
    print(f"  Total Account nodes:        {row['total']}")
    print(
        f"  With risk_score:            {row['has_risk']}  (avg: {row['avg_risk']:.6f})"
    )
    print(
        f"  With community_id:          {row['has_community']}  ({row['num_communities']} distinct communities)"
    )
    print(f"  With similarity_score:      {row['has_similarity']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from graph_feature_forge import inject_params

    inject_params()

    # --- Configuration from environment ---
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        print("ERROR: NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD are required.")
        sys.exit(1)

    print("=" * 60)
    print("  GDS Demo: PageRank + Louvain + Node Similarity")
    print("=" * 60)

    # --- Step 1: Connect ---
    print("\nStep 1/6: Connect to Neo4j GDS")
    gds = connect(neo4j_uri, neo4j_username, neo4j_password, neo4j_database)

    # --- Step 2: Project ---
    print("\nStep 2/6: Create graph projection")
    G = create_projection(gds)

    # --- Step 3: PageRank ---
    print("\nStep 3/6: PageRank → risk_score")
    run_pagerank(gds, G)

    # --- Step 4: Louvain ---
    print("\nStep 4/6: Louvain → community_id")
    run_louvain(gds, G)

    # --- Step 5: Node Similarity ---
    print("\nStep 5/6: Node Similarity → similarity_score")
    run_node_similarity(gds, G)

    # --- Step 6: Write back ---
    print("\nStep 6/6: Write properties to Account nodes")
    write_properties(gds, G)

    # --- Verify ---
    print("\nVerification:")
    verify_results(gds)

    # --- Cleanup ---
    G.drop()
    print(f"\nProjection '{PROJECTION_NAME}' dropped.")
    print("\nGDS Demo complete.")


if __name__ == "__main__":
    main()
