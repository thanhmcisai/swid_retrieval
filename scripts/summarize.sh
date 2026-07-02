#!/usr/bin/env bash
# Dump every number needed to fill the paper into one block (+ a log file).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python -m swid_retrieval.summarize 2>&1 | tee "$ROOT_PATH/results/full954_summary.txt"
