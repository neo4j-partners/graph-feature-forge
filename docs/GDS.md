# Graph Feature Engineering with Neo4j GDS

This tutorial walks through the graph projection and algorithm pipeline used in graph_feature_forge to produce machine learning features from a customer portfolio graph in Neo4j Aura.

The portfolio graph connects Customers to Accounts, Positions, Stocks, and Companies through typed relationships. Each relationship captures a concrete financial action: a customer *has* an account, an account *holds* a position, a position references a security. GDS algorithms operate on this topology to produce features that no tabular query can replicate, because the signal lives in the structure of connections, not in any single row.

## Prerequisites

- Neo4j Aura instance with the GDS plugin enabled
- Python `graphdatascience` library (`pip install graphdatascience`)
- Data loaded into Neo4j (see [Loading Data](#loading-data-into-neo4j) below)

```python
from graphdatascience import GraphDataScience

gds = GraphDataScience(
    "neo4j+s://your-aura-instance.databases.neo4j.io",
    auth=("neo4j", "your-password"),
    database="neo4j",
)
print(f"GDS version: {gds.version()}")
```

## The Graph Data Model

Seven node types and seven relationship types form the portfolio graph:

```
(Customer)-[:HAS_ACCOUNT]->(Account)-[:AT_BANK]->(Bank)
(Account)-[:HAS_POSITION]->(Position)-[:OF_SECURITY]->(Stock)-[:OF_COMPANY]->(Company)
(Account)-[:PERFORMS]->(Transaction)-[:BENEFITS_TO]->(Account)
```

The CSV source files that seed this graph:

| File | Node/Relationship | Key Columns |
|------|-------------------|-------------|
| `customers.csv` | `Customer` | `customer_id`, `risk_profile`, `annual_income`, `credit_score` |
| `accounts.csv` | `Account` | `account_id`, `account_type`, `balance` |
| `banks.csv` | `Bank` | `bank_id` |
| `companies.csv` | `Company` | `company_id`, `industry`, `sector`, `market_cap_billions` |
| `stocks.csv` | `Stock` | `stock_id`, `ticker`, `current_price` |
| `portfolio_holdings.csv` | `Position` | `holding_id`, `shares`, `current_value` |
| `transactions.csv` | `Transaction` | `transaction_id`, `amount`, `type` |

## Loading Data into Neo4j

The seeding pipeline creates uniqueness constraints, then writes nodes and relationships via batched UNWIND queries. Each node type gets a constraint on its key property, which also serves as the MERGE key during loading.

**Constraints:**

```cypher
CREATE CONSTRAINT customer_customer_id_unique IF NOT EXISTS
  FOR (n:Customer) REQUIRE n.customer_id IS UNIQUE;

CREATE CONSTRAINT account_account_id_unique IF NOT EXISTS
  FOR (n:Account) REQUIRE n.account_id IS UNIQUE;

-- (repeat for Bank, Company, Stock, Position, Transaction)
```

**Node loading pattern** (using Customer as an example):

```cypher
UNWIND $rows AS row
MERGE (n:Customer {customer_id: row.customer_id})
SET n.first_name = row.first_name,
    n.last_name = row.last_name,
    n.risk_profile = row.risk_profile,
    n.annual_income = row.annual_income,
    n.credit_score = row.credit_score
```

**Relationship loading pattern** (using HAS_ACCOUNT as an example):

```cypher
UNWIND $rows AS row
MATCH (src:Customer {customer_id: row.source_key})
MATCH (tgt:Account {account_id: row.target_key})
MERGE (src)-[r:HAS_ACCOUNT]->(tgt)
```

Both patterns use batch sizes of 100 rows per UNWIND call. At 778 nodes and ~890 relationships, the entire graph seeds in seconds.

See `src/graph_feature_forge/graph/seeding.py` for the full implementation.

## Step 1: Project the Graph

A GDS projection creates an in-memory copy of the subgraph that algorithms operate on. The projection selects which node labels and relationship types to include, and critically, how to treat relationship direction.

The portfolio graph projection includes five node labels and four relationship types. All relationships are projected as `UNDIRECTED` because the algorithms need to traverse influence in both directions: a Customer's risk profile should propagate through their Accounts to their Positions, and a Company's sector characteristics should flow back through Stocks to the Accounts that hold them.

```python
PROJECTION_NAME = "portfolio-graph"

NODE_LABELS = ["Customer", "Account", "Position", "Stock", "Company"]

RELATIONSHIP_SPEC = {
    "HAS_ACCOUNT": {"orientation": "UNDIRECTED"},
    "HAS_POSITION": {"orientation": "UNDIRECTED"},
    "OF_SECURITY": {"orientation": "UNDIRECTED"},
    "OF_COMPANY": {"orientation": "UNDIRECTED"},
}
```

Note what's excluded: `Bank`, `Transaction`, `AT_BANK`, `PERFORMS`, and `BENEFITS_TO`. The transaction subgraph is excluded from the base projection because transaction nodes vastly outnumber other entities and would dominate the embedding space. Bank nodes add little discriminative signal since multiple customers share the same bank. These could be added to a separate projection for transaction-specific analysis.

**Create the projection:**

```python
# Drop any existing projection with the same name
try:
    gds.graph.drop(gds.graph.get(PROJECTION_NAME))
except Exception:
    pass

G, _ = gds.graph.project(PROJECTION_NAME, NODE_LABELS, RELATIONSHIP_SPEC)

print(f"Nodes: {G.node_count()}, Relationships: {G.relationship_count()}")
```

### Including Enrichment Relationships

If the LLM enrichment pipeline has already run and written new relationship types to Neo4j (e.g., `INTERESTED_IN`, `SIMILAR_RISK_PROFILE`), these can be folded into the projection dynamically:

```python
# Discover enrichment relationship types from the audit log
enrichment_rel_types = ["INTERESTED_IN", "SIMILAR_RISK_PROFILE"]

relationship_spec = dict(RELATIONSHIP_SPEC)
for rel_type in enrichment_rel_types:
    relationship_spec[rel_type] = {"orientation": "UNDIRECTED"}

G, _ = gds.graph.project(PROJECTION_NAME, NODE_LABELS, relationship_spec)
```

This is how the pipeline compounds: each enrichment cycle adds relationships that subsequent GDS runs incorporate, producing richer embeddings and tighter community assignments.

## Step 2: FastRP Embeddings

FastRP (Fast Random Projection) generates a fixed-dimensional vector for each node based on its neighborhood structure. Two nodes with similar connectivity patterns (same banks, overlapping stock positions, shared counterparties) will have similar embeddings, even if they share no direct relationship.

```python
EMBEDDING_DIM = 128

fastrp_result = gds.fastRP.mutate(
    G,
    mutateProperty="fastrp_embedding",
    embeddingDimension=EMBEDDING_DIM,
    iterationWeights=[0.0, 1.0, 1.0, 0.8, 0.5],
    randomSeed=42,
)

print(f"FastRP: {fastrp_result['nodePropertiesWritten']} nodes embedded")
```

**Parameter choices:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `embeddingDimension` | 128 | Balances expressiveness with feature table size. Each customer becomes 128 numeric columns. |
| `iterationWeights` | `[0.0, 1.0, 1.0, 0.8, 0.5]` | Five iterations with decaying weights. The `0.0` first weight ignores self-loops. Weights decay at hops 4-5 so distant neighbors contribute less. |
| `randomSeed` | 42 | Reproducibility across runs. |

The `mutate` execution mode writes results to the in-memory projection only, not to Neo4j. This is faster than `write` mode and lets you run multiple algorithms on the same projection before persisting anything.

## Step 3: Louvain Community Detection

Louvain assigns each node a `community_id` based on modularity optimization. Nodes that are more densely connected to each other than to the rest of the graph end up in the same community. In a portfolio graph, this surfaces organic clusters: groups of customers who share banks, hold similar stocks, or have overlapping counterparty networks.

```python
louvain_result = gds.louvain.mutate(
    G,
    mutateProperty="community_id",
    maxLevels=10,
    maxIterations=10,
)

print(f"Communities: {louvain_result['communityCount']}")
print(f"Modularity: {louvain_result['modularity']:.4f}")
```

**Parameter choices:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `maxLevels` | 10 | Maximum hierarchical levels. Louvain recursively merges communities; 10 levels is generous for a graph of this size. |
| `maxIterations` | 10 | Iterations per level for modularity optimization. |

The `communityCount` in the result tells you how many distinct clusters the algorithm found. `modularity` ranges from -0.5 to 1.0; higher values indicate stronger community structure. For a well-connected portfolio graph, expect modularity in the 0.3-0.7 range.

## Step 4: PageRank (Centrality)

PageRank identifies the most structurally important nodes by propagating influence through the graph. In a portfolio context, a high PageRank score on a Customer node means that customer's accounts connect to many other accounts, positions, and companies, making them a hub in the financial network.

```python
pagerank_result = gds.pageRank.mutate(
    G,
    mutateProperty="risk_score",
    maxIterations=20,
    dampingFactor=0.85,
)

print(f"PageRank: {pagerank_result['nodePropertiesWritten']} nodes scored")
print(f"Iterations: {pagerank_result['ranIterations']}")
```

**Parameter choices:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `maxIterations` | 20 | Standard convergence ceiling. PageRank typically converges in fewer iterations. |
| `dampingFactor` | 0.85 | The classic value. At each step, 85% of a node's score propagates to neighbors; 15% distributes uniformly. |

The resulting `risk_score` is a relative measure: compare it across customers to find which accounts sit at the center of the network. Normalize before using as a feature if you need values between 0 and 1.

## Step 5: Node Similarity

Node Similarity computes pairwise Jaccard similarity between nodes based on their shared neighbors. Two Customer nodes that share many of the same Account-Position-Stock paths will have a high similarity score. This directly answers "which accounts share the most counterparties?"

Node Similarity differs from FastRP in an important way: FastRP produces per-node embeddings (one vector per customer), while Node Similarity produces per-pair scores (one score per customer pair). This means the output is a set of new relationships, not a node property.

```python
similarity_result = gds.nodeSimilarity.mutate(
    G,
    mutateRelationshipType="SIMILAR_TO",
    mutateProperty="similarity_score",
    similarityCutoff=0.1,
    topK=10,
)

print(f"Relationships created: {similarity_result['relationshipsWritten']}")
print(f"Similarity distribution: min={similarity_result['similarityDistribution']['min']:.4f}, "
      f"max={similarity_result['similarityDistribution']['max']:.4f}, "
      f"mean={similarity_result['similarityDistribution']['mean']:.4f}")
```

**Parameter choices:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `similarityCutoff` | 0.1 | Minimum Jaccard similarity to create a relationship. Filters out noise from barely-connected pairs. |
| `topK` | 10 | Maximum similar neighbors per node. Keeps the similarity graph sparse and the feature table manageable. |

### Alternative: kNN on FastRP Embeddings

If you've already computed FastRP embeddings, you can use kNN instead of Node Similarity to find similar nodes based on embedding distance rather than shared neighbors:

```python
knn_result = gds.knn.mutate(
    G,
    mutateRelationshipType="KNN_SIMILAR",
    mutateProperty="knn_score",
    nodeProperties=["fastrp_embedding"],
    topK=5,
    randomSeed=42,
    sampleRate=1.0,
    deltaThreshold=0.001,
)
```

kNN on embeddings captures structural similarity across multiple hops, while Jaccard Node Similarity captures direct neighbor overlap. Both are useful; the choice depends on whether you care about local or global similarity.

## Step 6: Write Properties Back to Neo4j

After all algorithms have run against the in-memory projection, write the computed properties back to Neo4j as persistent node properties.

```python
# Write node properties to Customer nodes
gds.graph.nodeProperties.write(
    G,
    node_properties=["fastrp_embedding", "community_id", "risk_score"],
    node_labels=["Customer"],
)

# Write similarity relationships (if computed)
gds.graph.relationship.write(
    G,
    relationship_type="SIMILAR_TO",
    relationship_property="similarity_score",
)

print("Properties and relationships written to Neo4j")
```

Writing only to `Customer` nodes is deliberate. The downstream ML pipeline builds a feature table keyed on `customer_id`, so properties on Account or Stock nodes aren't needed as direct features. If you need features at other granularities, specify additional labels in the `node_labels` list.

## Step 7: Drop the Projection

GDS projections consume memory. Drop them when you're done.

```python
G.drop()
```

## Putting It All Together

The complete pipeline as a single function:

```python
from graphdatascience import GraphDataScience

def compute_gds_features(
    uri: str,
    username: str,
    password: str,
    database: str = "neo4j",
    enrichment_rel_types: list[str] | None = None,
) -> dict:
    gds = GraphDataScience(uri, auth=(username, password), database=database)

    # --- Projection ---
    node_labels = ["Customer", "Account", "Position", "Stock", "Company"]
    relationship_spec = {
        "HAS_ACCOUNT": {"orientation": "UNDIRECTED"},
        "HAS_POSITION": {"orientation": "UNDIRECTED"},
        "OF_SECURITY": {"orientation": "UNDIRECTED"},
        "OF_COMPANY": {"orientation": "UNDIRECTED"},
    }
    for rel_type in enrichment_rel_types or []:
        relationship_spec[rel_type] = {"orientation": "UNDIRECTED"}

    try:
        gds.graph.drop(gds.graph.get("portfolio-graph"))
    except Exception:
        pass

    G, _ = gds.graph.project("portfolio-graph", node_labels, relationship_spec)
    print(f"Projected: {G.node_count()} nodes, {G.relationship_count()} rels")

    # --- FastRP ---
    gds.fastRP.mutate(
        G,
        mutateProperty="fastrp_embedding",
        embeddingDimension=128,
        iterationWeights=[0.0, 1.0, 1.0, 0.8, 0.5],
        randomSeed=42,
    )

    # --- Louvain ---
    louvain = gds.louvain.mutate(
        G, mutateProperty="community_id", maxLevels=10, maxIterations=10,
    )

    # --- PageRank ---
    gds.pageRank.mutate(
        G, mutateProperty="risk_score", maxIterations=20, dampingFactor=0.85,
    )

    # --- Node Similarity ---
    gds.nodeSimilarity.mutate(
        G,
        mutateRelationshipType="SIMILAR_TO",
        mutateProperty="similarity_score",
        similarityCutoff=0.1,
        topK=10,
    )

    # --- Write back ---
    gds.graph.nodeProperties.write(
        G,
        node_properties=["fastrp_embedding", "community_id", "risk_score"],
        node_labels=["Customer"],
    )
    gds.graph.relationship.write(
        G, relationship_type="SIMILAR_TO", relationship_property="similarity_score",
    )

    stats = {
        "node_count": G.node_count(),
        "relationship_count": G.relationship_count(),
        "community_count": louvain["communityCount"],
        "modularity": louvain["modularity"],
    }

    G.drop()
    return stats
```

## Feature Summary

After the pipeline runs, each Customer node carries these properties:

| Property | Type | Algorithm | Business Meaning |
|----------|------|-----------|------------------|
| `fastrp_embedding` | `float[128]` | FastRP | Structural fingerprint encoding the customer's neighborhood topology |
| `community_id` | `int` | Louvain | Which cluster of accounts behave similarly |
| `risk_score` | `float` | PageRank | How central this account is in the financial network |

And the graph contains new `SIMILAR_TO` relationships between Customer nodes:

| Relationship | Property | Algorithm | Business Meaning |
|--------------|----------|-----------|------------------|
| `SIMILAR_TO` | `similarity_score` | Node Similarity | Which accounts share the most counterparties (Jaccard index) |

These feed directly into the downstream ML feature table. The export pipeline reads Customer nodes via the Neo4j Spark Connector, explodes the 128-dimensional embedding into individual columns, and writes a Delta table with columns: `customer_id`, `annual_income`, `credit_score`, `risk_category`, `community_id`, `risk_score`, `fastrp_0` through `fastrp_127`.

## Source Files

| File | Role |
|------|------|
| `src/graph_feature_forge/ml/feature_engineering.py` | Core GDS projection and algorithm calls |
| `src/graph_feature_forge/graph/seeding.py` | Load data from Delta tables into Neo4j |
| `src/graph_feature_forge/graph_schema.py` | Node and relationship type definitions |
| `agent_modules/gds_fastrp_features.py` | End-to-end FastRP pipeline with holdout and AutoML |
| `agent_modules/gds_community_features.py` | Adds Louvain on top of FastRP, retrains and promotes |
| `agent_modules/ml_baseline_comparison.py` | Three-way comparison: tabular vs. FastRP vs. FastRP+Louvain |
