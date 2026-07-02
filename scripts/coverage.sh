#!/usr/bin/env bash
# Cross-check every paper table/figure against the produced run artifacts.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python -m swid_retrieval.coverage "$@" "$ROOT_PATH/paper_cea_submission.tex"
