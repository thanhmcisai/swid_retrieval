#!/usr/bin/env bash
# OPT-IN data prep (foreground): raw images → patches → split/IAWA/public → manifest.
#
# NON-DESTRUCTIVE: skip-if-exists by default; GBIF offline (cache-only). To
# regenerate without overwriting validated files, point ROOT_PATH / PSI_DIR at a
# fresh location, e.g.:
#   ROOT_PATH=$HOME/swid_repro FORCE_DATAPREP=1 bash swid_retrieval/scripts/dataprep.sh
#
# Stage gating (all default 1): RUN_DATAPREP_{CROP,SELECT,PUBLIC,SPLIT,IAWA,MANIFEST}
# GBIF_OFFLINE=0 enables live GBIF lookups (network). Logs to results/dataprep.log.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

mkdir -p "$ROOT_PATH/results"
LOG="$ROOT_PATH/results/dataprep.log"

echo "== dataprep config =="
echo "  ROOT_PATH=$ROOT_PATH  FORCE_DATAPREP=${FORCE_DATAPREP:-0}  GBIF_OFFLINE=${GBIF_OFFLINE:-1}"
echo "  stages: CROP=${RUN_DATAPREP_CROP:-1} SELECT=${RUN_DATAPREP_SELECT:-1} PUBLIC=${RUN_DATAPREP_PUBLIC:-1} SPLIT=${RUN_DATAPREP_SPLIT:-1} IAWA=${RUN_DATAPREP_IAWA:-1} MANIFEST=${RUN_DATAPREP_MANIFEST:-1}"

echo "== dataprep (tee -> $LOG) =="
python -m swid_retrieval.dataprep.build_all 2>&1 | tee "$LOG"
