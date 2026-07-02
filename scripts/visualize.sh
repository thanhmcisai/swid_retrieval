#!/usr/bin/env bash
# Render the paper figures (vector PDFs) from an existing run's per-scope results.
# Reads <run>/{paradigm,deployment,ce_train_robustness}/ and writes <run>/figures/.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python -m swid_retrieval.visualize "$@"
