#!/usr/bin/env bash
# Anti-bug diagnostics on the already-built full-954 cache (no re-extraction):
# reproduce id_only (must match published 24-gallery numbers), R@k curves,
# SC-URD raw-vs-centered, and same-domain sanity. ~5-10 min.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python -m swid_retrieval.diagnose 2>&1 | tee "$ROOT_PATH/results/diagnose_full954.log"
