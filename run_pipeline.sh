#!/usr/bin/env bash
#
# Run the full semantic-auth pipeline on Databricks.
#
# Usage:
#   ./run_pipeline.sh          # Run all steps (upload + load + seed + enrich)
#   ./run_pipeline.sh --seed   # Skip enrichment (upload + load + seed only)
#
set -euo pipefail

cd "$(dirname "$0")"

SEED_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --seed) SEED_ONLY=true ;;
        *) echo "Unknown option: $arg"; echo "Usage: $0 [--seed]"; exit 1 ;;
    esac
done

echo "=== Upload raw data to UC volume ==="
uv run python -m cli upload --data

echo ""
echo "=== Build and upload wheel + entry points ==="
uv run python -m cli upload --wheel
uv run python -m cli upload load_data.py
uv run python -m cli upload seed_neo4j.py
uv run python -m cli upload run_semantic_auth.py

echo ""
echo "=== Create Delta tables from CSVs ==="
uv run python -m cli submit load_data.py

echo ""
echo "=== Seed Neo4j from Delta tables + embeddings ==="
uv run python -m cli submit seed_neo4j.py

if [ "$SEED_ONLY" = false ]; then
    echo ""
    echo "=== Run enrichment pipeline ==="
    uv run python -m cli submit run_semantic_auth.py

    echo ""
    echo "=== Logs ==="
    uv run python -m cli logs
fi

echo ""
echo "=== Pipeline complete ==="
