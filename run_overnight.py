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

Set RUN_FINAL_COLAB_AUDIT=1 before invoking this module to run only the
inference-only final audit. Full-pipeline flags are ignored in that mode.
Set RUN_FINAL_SCURD_RETRAIN=1 to run only the isolated three-seed head rerun.
Set RUN_GALLERY_STUDY=1 to run only the isolated end-to-end gallery study.
Set RUN_GALLERY_DIAGNOSTICS=1 for inference-only gallery checkpoint diagnosis.
Set RUN_WOOD_EVIDENCE_STUDY=1 for the isolated local-evidence retrieval study.
"""

import os

# Set env BEFORE importing the package so config picks it up.
os.environ.setdefault("ROOT_PATH", "/content/drive/MyDrive/NCS")
_DEFAULTS = {
    # Protocol: full 954-species SWI gallery (cardinality-matched to CE).
    "GALLERY_SCOPE": "full_swi",
    "MIN_FULL_GALLERY_SPECIES": "900",
    "FULL954_CACHE_NAME": "embedding_cache_full954_v4_corrected_public.npz",
    "SOURCE_FULL954_CACHE_NAME": "embedding_cache_full954_v3.npz",
    "RUN_PUBLIC_DATAPREP": "1",
    "RUN_PUBLIC_CACHE_MIGRATION": "1",
    "FORCE_DATAPREP": "1",
    "FORCE_PUBLIC_EXPAND": "1",
    "EXCLUDE_WOODAUTH": "1",
    "APPLY_FSDM41_CORRECTION": "1",
    # Step 0: build cache + parallel image pre-warm (Drive is slow per-file).
    "RUN_BUILD_FULL954": "1",
    "FORCE_REBUILD_FULL954": "0",   # 0 -> skip if a valid 954 cache already exists
    "SAVE_PARTIAL": "1",            # resumable extraction
    "PRELOAD_IMAGE_CACHE": "1",
    "PRELOAD_IMAGE_SCOPE": "all",     # all | public | swi | none
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
    "RUN_HYPERPARAM_SELECTION": "1",      # meta-val-only SC-URD selection audit
    "SCURD_TRAIN_SEEDS": "42,43,44",
    "SCURD_TRAIN_EPOCHS": "20",
    "SCURD_TRAIN_EPISODES": "500",
    "SCURD_TRAIN_LR": "1e-3",
    "SCURD_TRAIN_LAMBDA_CONS": "0.5",
    "SCURD_FORCE_RETRAIN_SEEDS": "0",
    "FORCE_HYPERPARAM_SELECTION": "0",
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
    "RUN_CLASS_INCREMENTAL": "1",         # stronger CE adaptation control
    "RUN_SCURD_SELECTED_HPARAM_EVAL": "1",# evaluate meta-val selected SC-URD scorer
    "RUN_SCURD_TRAINING_ABLATION": "1",   # REV-1/2: lr + consistency ablations
    "RUN_OOD_WITHIN_SOURCE": "1",         # REV-4: source-controlled OOD protocol
    "RUN_CLASS_GEOMETRY": "1",            # RF/GEO: reference-matching geometry diagnostics
    "RUN_SUPCON_SEED_SENSITIVITY": "0",   # REV-3: opt-in, trains/extracts image backbones
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
    if os.environ.get("RUN_WOOD_EVIDENCE_STUDY", "0") == "1":
        if any(os.environ.get(flag, "0") == "1" for flag in (
                "RUN_GALLERY_DIAGNOSTICS", "RUN_GALLERY_STUDY", "RUN_REPAIR_PUBLIC_ROWS",
                "RUN_FINAL_SCURD_RETRAIN", "RUN_FINAL_COLAB_AUDIT")):
            raise ValueError("Run the wood-evidence study separately from other modes")
        from swid_retrieval import wood_evidence_experiment
        wood_evidence_experiment.run()
    elif os.environ.get("RUN_GALLERY_DIAGNOSTICS", "0") == "1":
        if any(os.environ.get(flag, "0") == "1" for flag in (
                "RUN_GALLERY_STUDY", "RUN_REPAIR_PUBLIC_ROWS",
                "RUN_FINAL_SCURD_RETRAIN", "RUN_FINAL_COLAB_AUDIT")):
            raise ValueError("Run gallery diagnostics separately from other modes")
        from swid_retrieval import gallery_diagnostics
        gallery_diagnostics.run()
    elif os.environ.get("RUN_GALLERY_STUDY", "0") == "1":
        if any(os.environ.get(flag, "0") == "1" for flag in (
                "RUN_REPAIR_PUBLIC_ROWS", "RUN_FINAL_SCURD_RETRAIN", "RUN_FINAL_COLAB_AUDIT")):
            raise ValueError("Run gallery study separately from repair and final-audit modes")
        from swid_retrieval import gallery_experiment
        gallery_experiment.run()
    elif os.environ.get("RUN_REPAIR_PUBLIC_ROWS", "0") == "1":
        if os.environ.get("RUN_FINAL_SCURD_RETRAIN", "0") == "1" or os.environ.get("RUN_FINAL_COLAB_AUDIT", "0") == "1":
            raise ValueError("Run public row repair separately from final audit/head training")
        from swid_retrieval.embeddings import repair_public_rows
        if os.environ.get("PUBLIC_REPAIR_DIAGNOSE_ONLY", "0") == "1":
            repair_public_rows.diagnose_duplicate_rows()
        else:
            repair_public_rows.run()
    elif os.environ.get("RUN_FINAL_SCURD_RETRAIN", "0") == "1":
        if os.environ.get("RUN_FINAL_COLAB_AUDIT", "0") == "1":
            raise ValueError("Choose only one of RUN_FINAL_SCURD_RETRAIN and RUN_FINAL_COLAB_AUDIT")
        from swid_retrieval import final_scurd_retrain
        final_scurd_retrain.run()
    elif os.environ.get("RUN_FINAL_COLAB_AUDIT", "0") == "1":
        from argparse import Namespace
        from pathlib import Path
        from swid_retrieval import final_colab_audit

        root = Path(os.environ["ROOT_PATH"])
        run_root = Path(os.environ.get(
            "FINAL_AUDIT_RUN_ROOT",
            root / "results" / "paper_reframe_full954_retrained_ce_corrected_public"))
        out = Path(os.environ.get("FINAL_AUDIT_OUT", run_root / "final_colab_audit"))
        cache_name = os.environ.get("FINAL_AUDIT_CACHE_NAME")
        cache = Path(cache_name) if cache_name else None
        if cache is not None and not cache.is_absolute():
            cache = root / cache
        args = Namespace(
            root=root, run_root=run_root, out=out,
            research_dir=None, cache=cache, exp4=None, ce_checkpoint=None,
            device=os.environ.get("DEVICE", "cuda"),
            batch_size=int(os.environ.get("FINAL_AUDIT_BATCH_SIZE", "64")),
            workers=int(os.environ.get("FINAL_AUDIT_WORKERS", "4")),
            gallery_repeats=int(os.environ.get("FINAL_AUDIT_GALLERY_REPEATS", "100")),
            ood_kshot_repeats=int(os.environ.get("FINAL_AUDIT_OOD_KSHOT_REPEATS", "50")),
            preflight=os.environ.get("FINAL_AUDIT_PREFLIGHT_ONLY", "0") == "1",
        )
        print(f"FINAL AUDIT ONLY: {out}", flush=True)
        final_colab_audit.run(args)
    else:
        print(f"ROOT_PATH={os.environ['ROOT_PATH']}  GALLERY_SCOPE={os.environ['GALLERY_SCOPE']}")
        from swid_retrieval import orchestrator
        run_dir = orchestrator.main()
        print(f"\n===== DONE. Results: {run_dir} =====")
