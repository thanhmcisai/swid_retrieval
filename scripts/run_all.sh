#!/usr/bin/env bash
# Full overnight run (foreground): preflight -> build cache -> deployment/id_only
# -> paradigm/full_swi -> review evidence -> edge -> ce_train/native experiments
# -> figures/export/coverage-ready artifacts. Logs to results/.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

mkdir -p "$ROOT_PATH/results"
LOG="$ROOT_PATH/results/full954_overnight.log"

echo "== preflight =="
python -m swid_retrieval.preflight    # aborts here (set -e) if anything is missing

echo "== run (tee -> $LOG) =="
python -m swid_retrieval.run_overnight 2>&1 | tee "$LOG"
