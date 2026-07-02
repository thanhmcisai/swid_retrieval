#!/usr/bin/env bash
# Step 0 only: pre-warm images + build the full-954 embedding cache (+ ce_train
# mask). Idempotent — skips if a valid cache already exists. Useful to run the
# heavy extraction once, separately from the (fast) evaluation passes.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python -c "from swid_retrieval.embeddings import build_full954 as b; b.main()"
