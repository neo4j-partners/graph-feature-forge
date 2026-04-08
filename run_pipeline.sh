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

usage() {
    echo "Usage: $0 [load|seed|enrich]"
    echo ""
    echo "Phases:"
    echo "  load     Upload raw data + create Delta tables from CSVs"
    echo "  seed     Seed Neo4j from Delta tables + embeddings"
    echo "  enrich   Run enrichment pipeline"
    echo ""
    echo "With no argument, all phases run in order: load → seed → enrich"
    exit 1
}

phase_load() {
    echo "=== Upload raw data to UC volume ==="
    uv run python -m cli upload --data

    echo ""
    echo "=== Build and upload wheel + entry point ==="
    uv run python -m cli upload --wheel
    uv run python -m cli upload load_data.py

    echo ""
    echo "=== Create Delta tables from CSVs ==="
    uv run python -m cli submit load_data.py
}

phase_seed() {
    echo "=== Build and upload wheel + entry point ==="
    uv run python -m cli upload --wheel
    uv run python -m cli upload seed_neo4j.py

    echo ""
    echo "=== Seed Neo4j from Delta tables + embeddings ==="
    uv run python -m cli submit seed_neo4j.py
}

phase_enrich() {
    echo "=== Build and upload wheel + entry point ==="
    uv run python -m cli upload --wheel
    uv run python -m cli upload run_semantic_auth.py

    echo ""
    echo "=== Run enrichment pipeline ==="
    uv run python -m cli submit run_semantic_auth.py --compute cluster

    echo ""
    echo "=== Logs ==="
    uv run python -m cli logs
}

PHASE="${1:-all}"

case "$PHASE" in
    load)    phase_load ;;
    seed)    phase_seed ;;
    enrich)  phase_enrich ;;
    all)
        phase_load
        echo ""
        phase_seed
        echo ""
        phase_enrich
        ;;
    *)       usage ;;
esac

echo ""
echo "=== Pipeline complete ==="
