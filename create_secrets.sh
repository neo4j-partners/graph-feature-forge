#!/usr/bin/env bash
#
# Provision a Databricks secret scope from .env values.
#
# Reads DATABRICKS_SECRET_SCOPE from .env, creates the scope (if it
# doesn't already exist), and stores each secret key's value from .env.
#
# Usage:
#   ./create_secrets.sh            # Use DATABRICKS_PROFILE from .env
#   ./create_secrets.sh <profile>  # Override profile
#
set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Parse a value from .env by key name
# ---------------------------------------------------------------------------
env_val() {
    local key=$1
    local line
    line="$(grep "^${key}=" .env 2>/dev/null | head -1)" || true
    [ -z "$line" ] && return 0
    echo "$line" | cut -d= -f2- | sed "s/^[\"']//;s/[\"']$//"
}

PROFILE="${1:-$(env_val DATABRICKS_PROFILE)}"
SCOPE="$(env_val DATABRICKS_SECRET_SCOPE)"

# Keys whose values are stored as Databricks secrets.
# Must match SECRET_KEYS in cli/__init__.py.
SECRET_KEYS=(NEO4J_URI NEO4J_USERNAME NEO4J_PASSWORD)

if [ -z "$SCOPE" ]; then
    echo "ERROR: DATABRICKS_SECRET_SCOPE not set in .env" >&2
    exit 1
fi

PROFILE_FLAG=""
if [ -n "${PROFILE:-}" ]; then
    PROFILE_FLAG="--profile ${PROFILE}"
fi

echo "Creating secret scope: $SCOPE"
# shellcheck disable=SC2086
databricks secrets create-scope "$SCOPE" $PROFILE_FLAG 2>/dev/null \
    || echo "  (scope already exists)"

for KEY in "${SECRET_KEYS[@]}"; do
    VALUE="$(env_val "$KEY")"
    if [ -z "$VALUE" ]; then
        echo "  WARNING: $KEY has no value in .env, skipping"
        continue
    fi
    echo "  Putting secret: $KEY"
    # shellcheck disable=SC2086
    databricks secrets put-secret "$SCOPE" "$KEY" \
        --string-value "$VALUE" $PROFILE_FLAG
done

echo "Done. Secrets stored in scope '$SCOPE'."
