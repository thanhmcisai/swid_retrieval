# -*- coding: utf-8 -*-
"""Pre-flight artifact check. Run before the overnight job so it fails in seconds,
not at 3am. Exits non-zero if anything required is missing.

    python -m swid_retrieval.preflight
"""

import json
import sys
from pathlib import Path

from . import config


def _engine_path(filename):
    """Resolve an engine script via the same logic the orchestrator uses."""
    from .orchestrator import _find_script
    key = {
        "variance_retrieval_evidence_colab.py": "VARIANCE_SCRIPT_PATH",
        "review_evidence_colab.py": "REVIEW_SCRIPT_PATH",
        "edge_deployment_proxy_colab.py": "EDGE_SCRIPT_PATH",
    }[filename]
    return _find_script(key, filename)


def main():
    R = config.ROOT_PATH
    config.configure_runtime(verbose=True)
    try:
        from .data import prepare_public_csvs
        prepare_public_csvs(R)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  public CSV preparation skipped during preflight: {exc}")

    checks = {
        "swi_manifest":      config.MANIFEST_PATH,
        "old emb cache":     config.SOURCE_EMB_CACHE_PATH,   # source of ID/OOD copy
        "exp4 cache (VN26)": R / config.EXP4_CACHE_NAME,
        "ID images csv":     config.ID_IMAGES_CSV,
        "OOD images csv":    config.OOD_IMAGES_CSV,
        "CE-Full ckpt":      config.CKPT_CE_FULL,
        "CE-Narrow ckpt":    config.CKPT_CE_NARROW,
        "ArcFace-557 ckpt":  config.CKPT_ARC_557,
        "ArcFace-954 ckpt":  config.CKPT_ARC_954,
        "ProtoNet ckpt":     config.CKPT_PROTO,
        "Var ViT-B ckpt":    config.CKPT_DIR / "metric_vit_base_patch16_224_arcface.pt",
        "Var RN50 ckpt":     config.CKPT_DIR / "metric_resnet50_arcface.pt",
        "Var SupCon ckpt":   config.CKPT_DIR / "metric_convnext_base_supcon.pt",
        "SC-URD ckpt":       config.SCURD_MAIN_CKPT,
        "SC-URD proj cache": config.SCURD_PROJ_CACHE,
        "SC-URD meta cache": config.RESEARCH_DIR / "urd_v2_meta_dinov2_embeddings_v2.npz",
        "old rq1 (sanity)":  config.RESULTS_DIR / "rq1_paradigm.json",
        "variance engine":   _engine_path("variance_retrieval_evidence_colab.py"),
        "review engine":     _engine_path("review_evidence_colab.py"),
        "edge engine":       _engine_path("edge_deployment_proxy_colab.py"),
    }

    print(f"ROOT_PATH = {R}")
    missing = [name for name, p in checks.items() if not p.exists()]
    for name, p in checks.items():
        print(f"  [{'OK ' if p.exists() else 'MISSING'}] {name:20s} {p}")

    # Sample image path from the manifest must resolve (Drive paths correct).
    img_ok = True
    try:
        m = json.load(open(config.MANIFEST_PATH))
        sample = m["meta-train"][0][0]
        img_ok = Path(sample).exists()
        print(f"\n  sample image: {sample}\n  exists: {img_ok}")
    except Exception as e:  # noqa: BLE001
        img_ok = False
        print(f"\n  could not read a sample image path: {e}")

    try:
        import torch
        print(f"  CUDA: {torch.cuda.is_available()} "
              f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    except Exception:
        print("  torch not importable")

    ok = (not missing) and img_ok
    if ok:
        print("\n==> READY")
    else:
        why = (f"missing: {missing}" if missing else "") + \
              ("" if img_ok else "  + sample image path does not resolve")
        print(f"\n==> NOT READY — {why}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
