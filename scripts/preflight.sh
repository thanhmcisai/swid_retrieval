#!/usr/bin/env bash
# Verify all artifacts exist before the long run. Exits non-zero if not ready.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python -m swid_retrieval.preflight
