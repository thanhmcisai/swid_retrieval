#!/usr/bin/env bash
# OPT-IN training pipeline (foreground): reproduce every validated checkpoint
# (MAIN backbones -> VARIANTS -> RESEARCH URD/SC-URD grid -> PHASE2 -> SC-URD seeds).
#
# NON-DESTRUCTIVE: skip-if-exists by default. To reproduce WITHOUT overwriting the
# released weights, point CKPT_DIR / RESULTS_DIR at a fresh directory, e.g.
#   CKPT_DIR=$ROOT_PATH/checkpoints_repro RESULTS_DIR=$ROOT_PATH/results/repro \
#     FORCE_RETRAIN=1 bash swid_retrieval/scripts/train.sh
#
# Stage gating (all default 1): RUN_TRAIN_{MAIN,VARIANTS,RESEARCH,PHASE2,SCURD_SEEDS}
# FAST=1 runs a 2-epoch smoke profile. Logs to results/train.log.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

mkdir -p "$ROOT_PATH/results"
LOG="$ROOT_PATH/results/train.log"

echo "== training config =="
echo "  CKPT_DIR=${CKPT_DIR:-$ROOT_PATH/checkpoints}"
echo "  RESULTS_DIR=${RESULTS_DIR:-$ROOT_PATH/results/paper_reframe}"
echo "  FORCE_RETRAIN=${FORCE_RETRAIN:-0}  FAST=${FAST:-0}"
echo "  stages: MAIN=${RUN_TRAIN_MAIN:-1} VARIANTS=${RUN_TRAIN_VARIANTS:-1} RESEARCH=${RUN_TRAIN_RESEARCH:-1} PHASE2=${RUN_TRAIN_PHASE2:-1} SEEDS=${RUN_TRAIN_SCURD_SEEDS:-1}"

echo "== train (tee -> $LOG) =="
python -m swid_retrieval.training.train_all 2>&1 | tee "$LOG"
