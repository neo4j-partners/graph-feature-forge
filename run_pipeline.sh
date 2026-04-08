#!/usr/bin/env bash
#
# Run the full graph-feature-forge pipeline on Databricks.
#
# Usage:
#   ./run_pipeline.sh          # Run all steps (upload + load + seed + enrich)
#   ./run_pipeline.sh --seed   # Skip enrichment (upload + load + seed only)
#
set -euo pipefail

cd "$(dirname "$0")"

usage() {
    echo "Usage: $0 [load|seed|enrich|gds]"
    echo ""
    echo "Phases:"
    echo "  load     Upload raw data + create Delta tables from CSVs"
    echo "  seed     Seed Neo4j from Delta tables + embeddings"
    echo "  enrich   Run enrichment pipeline"
    echo "  gds      Run GDS feature engineering (FastRP → Community → Baseline)"
    echo ""
    echo "With no argument, load + seed + enrich run in order."
    echo "The gds phase runs separately and requires a ML Runtime cluster."
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
    uv run python -m cli upload run_graph_feature_forge.py

    echo ""
    echo "=== Run enrichment pipeline ==="
    uv run python -m cli submit run_graph_feature_forge.py --compute cluster

    echo ""
    echo "=== Logs ==="
    uv run python -m cli logs
}

phase_gds() {
    echo "=== Build and upload wheel ==="
    uv run python -m cli upload --wheel

    echo ""
    echo "=== Upload GDS entry points ==="
    uv run python -m cli upload gds_fastrp_features.py
    uv run python -m cli upload gds_community_features.py
    uv run python -m cli upload gds_baseline_comparison.py

    echo ""
    echo "=== Submit GDS jobs (--no-wait) ==="
    echo ""
    echo "--- Step 1/3: GDS FastRP features ---"
    uv run python -m cli submit gds_fastrp_features.py --compute cluster --no-wait

    echo ""
    echo "--- Step 2/3: GDS Community features ---"
    uv run python -m cli submit gds_community_features.py --compute cluster --no-wait

    echo ""
    echo "--- Step 3/3: GDS Baseline comparison ---"
    uv run python -m cli submit gds_baseline_comparison.py --compute cluster --no-wait

    echo ""
    echo "============================================================"
    echo "  All GDS jobs submitted. They will run sequentially on the"
    echo "  cluster (each waits for its predecessor)."
    echo ""
    echo "  To check progress:"
    echo "    uv run python -m cli logs            # latest run"
    echo "    uv run python -m cli logs <run_id>   # specific run"
    echo ""
    echo "  What to look for:"
    echo "    - 'Step 5/5' in gds_fastrp_features = FastRP pipeline done"
    echo "    - 'Step 6/6' in gds_community_features = Community pipeline done"
    echo "    - 'Step 3/3' in gds_baseline_comparison = Baseline done"
    echo "    - 'Result: SUCCESS' = job completed successfully"
    echo "============================================================"
}

PHASE="${1:-all}"

case "$PHASE" in
    load)    phase_load ;;
    seed)    phase_seed ;;
    enrich)  phase_enrich ;;
    gds)     phase_gds ;;
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
