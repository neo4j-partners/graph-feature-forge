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
# JSON encoder
# ---------------------------------------------------------------------------


class _ReportEncoder(json.JSONEncoder):
    """Handle Neo4j date types and numpy scalars for JSON serialization."""

    def default(self, obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if hasattr(obj, "item"):
            return obj.item()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Fraud detection functions
# ---------------------------------------------------------------------------


def detect_suspicious_communities(gds):
    """Find communities where 3+ accounts share the same bank.

    After Louvain assigns community_id to Account nodes, this query
    identifies communities where multiple accounts cluster at a single
    bank — a hallmark of coordinated fraud rings.
    """
    result = gds.run_cypher("""
        MATCH (a:Account)-[:AT_BANK]->(b:Bank)
        WHERE a.community_id IS NOT NULL
        WITH a.community_id AS community, b.bank_id AS bank_id, b.name AS bank_name,
             collect({
                 account_id: a.account_id,
                 risk_score: a.risk_score,
                 similarity_score: a.similarity_score
             }) AS accounts
        WHERE size(accounts) >= 3
        RETURN community, bank_id, bank_name, size(accounts) AS account_count, accounts
        ORDER BY account_count DESC
    """)
    communities = []
    for _, row in result.iterrows():
        communities.append({
            "community_id": int(row["community"]),
            "bank_id": row["bank_id"],
            "bank_name": row["bank_name"],
            "account_count": int(row["account_count"]),
            "accounts": [dict(a) for a in row["accounts"]],
        })
    print(f"  Found {len(communities)} suspicious communities (3+ accounts at same bank)")
    for c in communities:
        print(
            f"    Community {c['community_id']}: "
            f"{c['account_count']} accounts at {c['bank_name']}"
        )
    return communities


def detect_structuring(gds):
    """Find accounts with 3+ transactions in the $8k-$10k structuring range.

    Currency structuring — keeping transactions just under the $10,000
    reporting threshold — is a classic money-laundering signal.
    """
    result = gds.run_cypher("""
        MATCH (a:Account)-[:PERFORMS]->(t:Transaction)
        WHERE t.amount >= 8000 AND t.amount < 10000
        WITH a, count(t) AS structuring_count,
             round(avg(t.amount), 2) AS avg_amount,
             round(sum(t.amount), 2) AS total_amount,
             min(toString(t.transaction_date)) AS first_date,
             max(toString(t.transaction_date)) AS last_date
        WHERE structuring_count >= 3
        RETURN a.account_id AS account_id,
               a.risk_score AS risk_score,
               a.community_id AS community_id,
               structuring_count,
               avg_amount,
               total_amount,
               first_date,
               last_date
        ORDER BY structuring_count DESC
    """)
    accounts = []
    for _, row in result.iterrows():
        accounts.append({
            "account_id": row["account_id"],
            "risk_score": float(row["risk_score"]) if row["risk_score"] is not None else None,
            "community_id": int(row["community_id"]) if row["community_id"] is not None else None,
            "structuring_count": int(row["structuring_count"]),
            "avg_amount": float(row["avg_amount"]),
            "total_amount": float(row["total_amount"]),
            "date_range": {
                "first": str(row["first_date"]),
                "last": str(row["last_date"]),
            },
        })
    print(f"  Found {len(accounts)} accounts with structuring patterns (3+ txns in $8k-$10k)")
    for a in accounts:
        print(
            f"    {a['account_id']}: {a['structuring_count']} txns, "
            f"avg ${a['avg_amount']:,.2f}, community {a['community_id']}"
        )
    return accounts


def detect_circular_flows(gds):
    """Detect circular money flows via structuring transactions.

    Returns directed edges (from -> to with counts) and ring membership —
    accounts that both send and receive structuring-amount transactions
    within the same Louvain community.
    """
    edges = gds.run_cypher("""
        MATCH (s:Account)-[:PERFORMS]->(t:Transaction)-[:BENEFITS_TO]->(r:Account)
        WHERE t.amount >= 8000 AND t.amount < 10000
          AND s.community_id IS NOT NULL
          AND s.community_id = r.community_id
        RETURN s.community_id AS community,
               s.account_id AS from_account,
               r.account_id AS to_account,
               count(t) AS txn_count,
               round(sum(t.amount), 2) AS total_amount
        ORDER BY community, from_account
    """)

    rings = gds.run_cypher("""
        MATCH (s:Account)-[:PERFORMS]->(t:Transaction)-[:BENEFITS_TO]->(r:Account)
        WHERE t.amount >= 8000 AND t.amount < 10000
          AND s.community_id IS NOT NULL
          AND s.community_id = r.community_id
        WITH s.community_id AS community,
             collect(DISTINCT s.account_id) AS senders,
             collect(DISTINCT r.account_id) AS receivers
        WITH community,
             [x IN senders WHERE x IN receivers] AS ring_members
        WHERE size(ring_members) >= 3
        RETURN community, ring_members, size(ring_members) AS ring_size
    """)

    edge_list = []
    for _, row in edges.iterrows():
        edge_list.append({
            "community_id": int(row["community"]),
            "from_account": row["from_account"],
            "to_account": row["to_account"],
            "txn_count": int(row["txn_count"]),
            "total_amount": float(row["total_amount"]),
        })

    ring_list = []
    for _, row in rings.iterrows():
        ring_list.append({
            "community_id": int(row["community"]),
            "ring_members": list(row["ring_members"]),
            "ring_size": int(row["ring_size"]),
        })

    print(f"  Found {len(ring_list)} circular flow rings across {len(edge_list)} directed edges")
    for r in ring_list:
        print(
            f"    Community {r['community_id']}: "
            f"{r['ring_size']}-account ring — {r['ring_members']}"
        )

    return {"edges": edge_list, "rings": ring_list}


def detect_coordinated_positions(gds):
    """Find stocks held by 3+ accounts in the same Louvain community.

    Coordinated positions in low-cap stocks across clustered accounts
    may indicate pump-and-dump schemes.
    """
    result = gds.run_cypher("""
        MATCH (a:Account)-[:HAS_POSITION]->(:Position)-[:OF_SECURITY]->(s:Stock)
              -[:OF_COMPANY]->(co:Company)
        WHERE a.community_id IS NOT NULL
        WITH a.community_id AS community, s.stock_id AS stock_id, s.ticker AS ticker,
             co.name AS company_name, s.market_cap_billions AS market_cap,
             collect(DISTINCT a.account_id) AS holders
        WHERE size(holders) >= 3
        RETURN community, stock_id, ticker, company_name, market_cap,
               holders, size(holders) AS holder_count
        ORDER BY market_cap ASC
    """)
    positions = []
    for _, row in result.iterrows():
        positions.append({
            "community_id": int(row["community"]),
            "stock_id": row["stock_id"],
            "ticker": row["ticker"],
            "company_name": row["company_name"],
            "market_cap_billions": float(row["market_cap"]),
            "holders": list(row["holders"]),
            "holder_count": int(row["holder_count"]),
        })
    print(f"  Found {len(positions)} coordinated positions (3+ accounts, same community)")
    for p in positions:
        print(
            f"    {p['ticker']} ({p['company_name']}): "
            f"{p['holder_count']} holders in community {p['community_id']}, "
            f"mcap ${p['market_cap_billions']:.2f}B"
        )
    return positions


def build_combined_risk_report(gds, structuring, circular_flows, positions):
    """Aggregate all fraud signals per account with customer details."""
    flagged = {}

    for s in structuring:
        aid = s["account_id"]
        flagged.setdefault(aid, [])
        flagged[aid].append(
            f"structuring: {s['structuring_count']} txns, avg ${s['avg_amount']:,.2f}"
        )

    for ring in circular_flows["rings"]:
        for aid in ring["ring_members"]:
            flagged.setdefault(aid, [])
            flagged[aid].append(f"circular_flow: {ring['ring_size']}-account ring")

    for pos in positions:
        for aid in pos["holders"]:
            if aid in flagged:
                flagged[aid].append(
                    f"coordinated_position: {pos['ticker']} "
                    f"(${pos['market_cap_billions']:.2f}B mcap)"
                )

    if not flagged:
        print("  No flagged accounts")
        return []

    account_ids = list(flagged.keys())
    details = gds.run_cypher(
        """
        MATCH (c:Customer)-[:HAS_ACCOUNT]->(a:Account)
        WHERE a.account_id IN $ids
        RETURN c.customer_id AS customer_id,
               c.first_name + ' ' + c.last_name AS name,
               c.annual_income AS annual_income,
               c.credit_score AS credit_score,
               a.account_id AS account_id,
               a.account_type AS account_type,
               a.risk_score AS risk_score,
               a.community_id AS community_id,
               a.similarity_score AS similarity_score
        ORDER BY a.risk_score DESC
        """,
        params={"ids": account_ids},
    )

    report = []
    for _, row in details.iterrows():
        aid = row["account_id"]
        report.append({
            "customer_id": row["customer_id"],
            "customer_name": row["name"],
            "annual_income": float(row["annual_income"]) if row["annual_income"] is not None else None,
            "credit_score": int(row["credit_score"]) if row["credit_score"] is not None else None,
            "account_id": aid,
            "account_type": row["account_type"],
            "risk_score": float(row["risk_score"]) if row["risk_score"] is not None else None,
            "community_id": int(row["community_id"]) if row["community_id"] is not None else None,
            "similarity_score": float(row["similarity_score"]) if row["similarity_score"] is not None else None,
            "signals": flagged.get(aid, []),
            "signal_count": len(flagged.get(aid, [])),
        })

    customers = {r["customer_id"] for r in report}
    print(f"  {len(report)} flagged accounts across {len(customers)} customers")
    for r in report:
        print(
            f"    {r['customer_id']} ({r['customer_name']}): "
            f"{r['account_id']} — {r['signal_count']} signals, "
            f"risk={r['risk_score']}"
        )

    return report


def write_fraud_report(report_data, path):
    """Write the fraud detection report to JSON."""
    with open(path, "w") as f:
        json.dump(report_data, f, indent=2, cls=_ReportEncoder)
    print(f"  Report written to {path}")


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

    # --- Cleanup projection (no longer needed for Cypher queries) ---
    G.drop()
    print(f"\nProjection '{PROJECTION_NAME}' dropped.")

    # ================================================================
    # Fraud Detection Report
    # ================================================================
    print("\n" + "=" * 60)
    print("  Fraud Detection Report")
    print("=" * 60)

    print("\nStep 7: Suspicious communities")
    communities = detect_suspicious_communities(gds)

    print("\nStep 8: Structuring detection")
    structuring = detect_structuring(gds)

    print("\nStep 9: Circular flow detection")
    circular = detect_circular_flows(gds)

    print("\nStep 10: Coordinated positions")
    positions = detect_coordinated_positions(gds)

    print("\nStep 11: Combined risk report")
    combined = build_combined_risk_report(gds, structuring, circular, positions)

    # --- Assemble and write JSON report ---
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "suspicious_communities": len(communities),
            "structuring_accounts": len(structuring),
            "circular_flow_rings": len(circular["rings"]),
            "coordinated_position_groups": len(positions),
            "total_flagged_accounts": len(combined),
            "total_flagged_customers": len({r["customer_id"] for r in combined}),
        },
        "suspicious_communities": communities,
        "structuring_detection": structuring,
        "circular_flows": circular,
        "coordinated_positions": positions,
        "combined_risk_report": combined,
    }

    print("\nWriting report:")
    write_fraud_report(report, REPORT_PATH)

    print("\nGDS Demo + Fraud Detection complete.")


if __name__ == "__main__":
    main()
