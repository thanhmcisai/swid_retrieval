# -*- coding: utf-8 -*-
"""Opt-in TRAINING subpackage — reproduces the checkpoints the eval consumes.

Lifts the validated training code from smartwoodid_experiments_full.py (which is
not importable: it has !pip / google.colab magics) into clean modules that reuse
the eval package's data/models/config/scurd. NOT called by the eval orchestrator.

Run once (heavy GPU) with:  python -m swid_retrieval.training.train_all
or:  bash swid_retrieval/scripts/train.sh

IMPORTANT: retraining is non-deterministic across GPUs, so new weights differ from
the validated checkpoints that produced the paper numbers. Default FORCE_RETRAIN=0
+ skip-if-exists; point CKPT_DIR/RESEARCH at a FRESH dir to reproduce without
overwriting the validated .pt files.
"""
