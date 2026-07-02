# -*- coding: utf-8 -*-
"""Foreground end-to-end runner for the SWID retrieval submission.

Runs the whole suite synchronously (no background): build the full-954 cache,
run the deployment/id_only experiments, run the matched-954 paradigm pass, run
CE-train robustness, native appendix experiments, review evidence, edge proxy,
figures, export, and cardinality sanity. One run produces every table/figure
artifact used by the submission.

Usage on Colab (after mounting Drive; swid_retrieval/ lives under ROOT_PATH):
    %cd /content/drive/MyDrive/NCS
    !python -u -m swid_retrieval.run_overnight 2>&1 | tee results/full954_overnight.log
  or simply:
    !bash swid_retrieval/scripts/run_all.sh

Re-run after a disconnect: the cache build skips (idempotent), trained SC-URD
seeds skip, and FULL954_RUN_STAMP is fixed so outputs resume in the same folder.
"""

import os

# Set env BEFORE importing the package so config picks it up.
os.environ.setdefault("ROOT_PATH", "/content/drive/MyDrive/NCS")
_DEFAULTS = {
    # Protocol: full 954-species SWI gallery (cardinality-matched to CE).
    "GALLERY_SCOPE": "full_swi",
    "MIN_FULL_GALLERY_SPECIES": "900",
    # Step 0: build cache + parallel image pre-warm (Drive is slow per-file).
    "RUN_BUILD_FULL954": "1",
    "FORCE_REBUILD_FULL954": "0",   # 0 -> skip if a valid 954 cache already exists
    "SAVE_PARTIAL": "1",            # resumable extraction
    "PRELOAD_IMAGE_CACHE": "1",
    "PRELOAD_ALL_IMAGES": "1",      # SWI + public ID/OOD image CSVs
    "IMAGE_CACHE_DIR": "/content/cache_images",
    "PRELOAD_WORKERS": "16",
    # Engine passes: deployment/id_only, paradigm/full_swi, and ce_train robustness
    # are selected inside orchestrator.py.
    "RUN_MAP": "1",
    "RUN_HEADLINE_RECOMPUTE": "1",        # RQ1 native/prototype/R@1 + OOD + E3 retention
    "RUN_GALLERY_RESAMPLING": "1",        # Exp1B K-shot variance
    "RUN_REVIEWER_GAP_FULL_GALLERY": "1", # operating-point / FPR95-CI / OOD-by-source
    "RUN_RQ5_FULL_GALLERY": "1",          # backbone/loss + SC-URD memory modes
    "RUN_SCURD_SEED_SENSITIVITY": "1",
    "RUN_TRAIN_SCURD_SEEDS": "1",         # train seeds 42/43/44 (skip if present)
    "SCURD_TRAIN_SEEDS": "42,43,44",
    "SCURD_TRAIN_EPOCHS": "20",
    "SCURD_TRAIN_EPISODES": "500",
    "SCURD_FORCE_RETRAIN_SEEDS": "0",
    "N_GALLERY_REPEATS": "100",
    "N_BOOT": "2000",
    # Steps 2/3/5.
    "RUN_REVIEW_TAXONOMY_FULL_GALLERY": "1",
    "RUN_REVIEW_TAXONOMY_DEPLOYMENT": "1",
    "RUN_INTERPRETABILITY": "1",
    "RUN_EDGE_PROXY": "1",
    "RUN_CPU_PROXY": "1",                 # paper table reports CPU proxy rows
    "RUN_CE_TRAIN_ROBUSTNESS": "1",       # ce_train pass: full gallery-dependent suite
    "RUN_HEAVY": "1",                     # paper cost tables require CE-finetune + costs
    "RUN_SCURD_BACKBONE": "1",            # paper tab:scurd_backbone
    # Overnight operation.
    "STRICT_SANITY": "0",                 # warn (don't crash) if the gate trips
    "FULL954_RUN_STAMP": "overnight",     # fixed run dir -> resume on re-run
    "DEVICE": "cuda",
    "SEED": "42",
    # Compatibility knobs for legacy engines/caches. The packaged cache builder
    # is simpler, but these are harmless and keep old copied engines in reuse mode.
    "STRICT_CACHE_META": "1",
    "ACCEPT_LEGACY_CACHE_WITHOUT_META": "1",
    "IGNORE_MTIME_IN_CACHE_META": "1",
    "FULL_ARTIFACT_HASH": "0",
    "RECOMPUTE_CE_FINETUNE_RQ3": "0",
    "RECOMPUTE_JOINT_RETRAIN_COST": "0",
    # Timing/cost sample sizes used in the submission tables.
    "RQ3_ENROLL_BENCH_IMAGES": "64",
    "RQ3_INFER_BENCH_IMAGES": "64",
    "RQ3_INFER_SEARCH_ITERS": "10",
    "PRELOAD_TENSORS": "1",
    "SCURD_MAIN_MODE": "centered",
}
for k, v in _DEFAULTS.items():
    os.environ.setdefault(k, v)

if __name__ == "__main__":
    import torch
    print(f"CUDA: {torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    print(f"ROOT_PATH={os.environ['ROOT_PATH']}  GALLERY_SCOPE={os.environ['GALLERY_SCOPE']}")
    from swid_retrieval import orchestrator
    run_dir = orchestrator.main()
    print(f"\n===== DONE. Results: {run_dir} =====")
