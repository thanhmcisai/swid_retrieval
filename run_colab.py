# -*- coding: utf-8 -*-
"""One-cell Colab entry point for the complete SWID retrieval run.

Usage in a Colab cell (after mounting Drive and `cd`-ing to the repo that holds
the swid_retrieval/ package + the *_colab.py scripts):

    from google.colab import drive; drive.mount("/content/drive")
    %cd /content/drive/MyDrive/NCS          # or wherever the repo lives
    import swid_retrieval.run_colab          # runs deployment, paradigm, native, figures/export

Or set overrides first:

    import os
    os.environ["ROOT_PATH"] = "/content/drive/MyDrive/NCS"
    os.environ["RUN_BUILD_FULL954"] = "1"    # 0 to reuse an existing 954 cache
    from swid_retrieval import orchestrator; orchestrator.main()
"""

import os

os.environ.setdefault("ROOT_PATH", "/content/drive/MyDrive/NCS")
os.environ.setdefault("GALLERY_SCOPE", "full_swi")
os.environ.setdefault("MIN_FULL_GALLERY_SPECIES", "900")
os.environ.setdefault("RUN_HEAVY", "1")
os.environ.setdefault("RUN_SCURD_BACKBONE", "1")
os.environ.setdefault("RUN_INTERPRETABILITY", "1")
os.environ.setdefault("RUN_CPU_PROXY", "1")
os.environ.setdefault("RUN_CLASS_INCREMENTAL", "1")
os.environ.setdefault("RUN_SCURD_SELECTED_HPARAM_EVAL", "1")
os.environ.setdefault("PRELOAD_IMAGE_CACHE", "1")
os.environ.setdefault("PRELOAD_IMAGE_SCOPE", "all")
os.environ.setdefault("PRELOAD_ALL_IMAGES", "1")
os.environ.setdefault("IMAGE_CACHE_DIR", "/content/cache_images")
os.environ.setdefault("PRELOAD_WORKERS", "16")
os.environ.setdefault("SEED", "42")
os.environ.setdefault("STRICT_CACHE_META", "1")
os.environ.setdefault("ACCEPT_LEGACY_CACHE_WITHOUT_META", "1")
os.environ.setdefault("IGNORE_MTIME_IN_CACHE_META", "1")
os.environ.setdefault("FULL_ARTIFACT_HASH", "0")
os.environ.setdefault("RECOMPUTE_CE_FINETUNE_RQ3", "0")
os.environ.setdefault("RECOMPUTE_JOINT_RETRAIN_COST", "0")
os.environ.setdefault("RQ3_ENROLL_BENCH_IMAGES", "64")
os.environ.setdefault("RQ3_INFER_BENCH_IMAGES", "64")
os.environ.setdefault("RQ3_INFER_SEARCH_ITERS", "10")
os.environ.setdefault("PRELOAD_TENSORS", "1")
os.environ.setdefault("SCURD_MAIN_MODE", "centered")
os.environ.setdefault("SCURD_TRAIN_LR", "1e-3")
os.environ.setdefault("SCURD_TRAIN_LAMBDA_CONS", "0.5")
os.environ.setdefault("RUN_HYPERPARAM_SELECTION", "1")
os.environ.setdefault("FORCE_HYPERPARAM_SELECTION", "0")

from . import orchestrator  # noqa: E402

if __name__ == "__main__" or os.environ.get("SWID_AUTORUN", "1") == "1":
    orchestrator.main()
