#!/usr/bin/env bash
# Shared setup for the step scripts: locate the repo root (parent of
# swid_retrieval/) so the package is importable, and default ROOT_PATH to it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # swid_retrieval/scripts
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"                  # dir containing swid_retrieval/
cd "$REPO_ROOT"
export ROOT_PATH="${ROOT_PATH:-$REPO_ROOT}"
export PYTHONUNBUFFERED=1
echo "REPO_ROOT=$REPO_ROOT  ROOT_PATH=$ROOT_PATH"
