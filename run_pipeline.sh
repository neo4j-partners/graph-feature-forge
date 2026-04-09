#!/usr/bin/env bash
#
# Run the full graph_feature_forge pipeline on Databricks.
#
# Usage:
#   ./run_pipeline.sh          # Run all steps (upload + load + seed + enrich)
#   ./run_pipeline.sh --seed   # Skip enrichment (upload + load + seed only)
#
set -euo pipefail

cd "$(dirname "$0")"

# Load .env so all phases pick up DATABRICKS_PROFILE, etc.
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

usage() {
    echo "Usage: $0 [load|seed|enrich|gds|html]"
    echo ""
    echo "Phases:"
    echo "  load     Upload raw data + create Delta tables from CSVs"
    echo "  seed     Seed Neo4j from Delta tables + embeddings"
    echo "  enrich   Run enrichment pipeline"
    echo "  gds      Run GDS feature engineering (FastRP → Community → Baseline)"
    echo "  gds_demo Run GDS demo (PageRank + Louvain + Node Similarity)"
    echo "  html     Generate HTML documents + embeddings via LLM endpoint"
    echo "  clean    Wipe UC volume, workspace dir, and job runs"
    echo ""
    echo "With no argument, load + seed + enrich run in order."
    echo "The gds phase runs separately and requires a ML Runtime cluster."
    echo "The html phase generates documents on-cluster using LLM/embedding endpoints."
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
    echo "=== Submit GDS jobs (sequential) ==="
    echo ""
    echo "--- Step 1/3: GDS FastRP features ---"
    uv run python -m cli submit gds_fastrp_features.py --compute cluster

    echo ""
    echo "--- Step 2/3: GDS Community features ---"
    uv run python -m cli submit gds_community_features.py --compute cluster

    echo ""
    echo "--- Step 3/3: GDS Baseline comparison ---"
    uv run python -m cli submit gds_baseline_comparison.py --compute cluster --no-wait

    echo ""
    echo "============================================================"
    echo "  FastRP and Community jobs ran sequentially."
    echo "  Baseline comparison submitted (--no-wait)."
    echo ""
    echo "  To check baseline progress:"
    echo "    uv run python -m cli logs            # latest run"
    echo "    uv run python -m cli logs <run_id>   # specific run"
    echo "============================================================"
}

phase_gds_demo() {
    echo "=== Build and upload wheel ==="
    uv run python -m cli upload --wheel

    echo ""
    echo "=== Upload GDS demo script ==="
    uv run python -m cli upload gds_demo.py

    echo ""
    echo "=== Run GDS demo ==="
    uv run python -m cli submit gds_demo.py --compute cluster

    echo ""
    echo "=== Logs ==="
    uv run python -m cli logs
}

phase_clean() {
    echo "=== Wipe UC volume ==="
    export DATABRICKS_CONFIG_PROFILE="${DATABRICKS_PROFILE}"
    uv run python -c "
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
path = '/Volumes/graph_feature_forge/enrichment/source-data'
count = 0
def delete_recursive(dir_path):
    global count
    for f in w.files.list_directory_contents(dir_path):
        if f.is_directory:
            delete_recursive(f.path)
            w.files.delete_directory(f.path)
        else:
            w.files.delete(f.path)
        print(f'  Deleted: {f.path}')
        count += 1
delete_recursive(path)
print(f'  Removed {count} items from volume')
"

    echo ""
    echo "=== Clean workspace dir + job runs ==="
    uv run python -m cli clean
}

phase_html() {
    echo "=== Build and upload wheel + entry point ==="
    uv run python -m cli upload --wheel
    uv run python -m cli upload generate_html.py

    echo ""
    echo "=== Generate HTML documents + embeddings ==="
    uv run python -m cli submit generate_html.py

    echo ""
    echo "=== Logs ==="
    uv run python -m cli logs
}

PHASE="${1:-all}"

case "$PHASE" in
    load)    phase_load ;;
    seed)    phase_seed ;;
    enrich)  phase_enrich ;;
    gds)      phase_gds ;;
    gds_demo) phase_gds_demo ;;
    html)     phase_html ;;
    clean)   phase_clean ;;
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
