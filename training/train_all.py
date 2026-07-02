# -*- coding: utf-8 -*-
"""End-to-end training orchestrator for the SmartWoodID retrieval paper.

Reproduces every validated checkpoint the eval loads, in dependency order:

  MAIN      CE-Full, CE-Narrow, ArcFace-557, ProtoNet-557, ArcFace-954
  VARIANTS  ViT-B/ArcFace, ResNet50/ArcFace, ConvNeXt/SupCon
  RESEARCH  DINOv2 meta-embeddings → URD-v2 grid → SC-URD residual/anchor/teacher/
            oodcons/train-length → B1 density gate
  PHASE2    SC-URD cross-domain fine-tune (4 configs, from scurd_r01_e10)
  SEEDS     scurd_r01_e20_seed{42,43,44} via the validated engine

OPT-IN and NON-DESTRUCTIVE by design:
  * Not imported by the eval orchestrator — run explicitly via scripts/train.sh or
    `python -m swid_retrieval.training.train_all`.
  * Every step is skip-if-exists; FORCE_RETRAIN=1 overrides.
  * Retraining is NOT bit-identical to the released weights (GPU/cuDNN nondeterminism).
    To reproduce WITHOUT clobbering the validated checkpoints, point CKPT_DIR /
    RESULTS_DIR at a fresh directory before running. The released numbers come from
    the shipped .pt files; this pipeline documents how they were produced.

Stages are gated by env flags RUN_TRAIN_{MAIN,VARIANTS,RESEARCH,PHASE2,SCURD_SEEDS}.
"""

import numpy as np
import torch

from .. import config as C
from ..data import load_swi_manifest, full_swi_items, preload_image_cache
from . import config as TC
from . import train_backbones as B
from . import train_research as R
from . import meta_embeddings as ME


def _device():
    dev = C.resolve_device()
    print(f"Device: {dev} | FORCE_RETRAIN={TC.FORCE_RETRAIN} | FAST={TC.FAST}")
    return dev


def _load_dinov2(device):
    from torchvision import transforms
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to(device)
    tf = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, tf


# ── Stages ────────────────────────────────────────────────────────────────
def run_main(cfg, manifest, device, force):
    print("\n" + "=" * 70 + "\nSTAGE: MAIN backbones\n" + "=" * 70)
    B.train_ce_classifier(cfg, manifest, device, force=force)
    B.train_ce_narrow(cfg, manifest, device, force=force)
    import dataclasses
    arc_cfg = dataclasses.replace(cfg, backbone="convnext_base", loss="arcface")
    B.train_metric_model(arc_cfg, manifest, device,
                         ckpt_name="metric_convnext_base_arcface.pt", force=force)
    proto_cfg = dataclasses.replace(cfg, backbone="convnext_base", loss="protonet")
    B.train_metric_model(proto_cfg, manifest, device,
                         ckpt_name="metric_convnext_base_protonet_v2.pt", force=force)
    B.train_arcface_954(cfg, manifest, device, force=force)


def run_variants(cfg, manifest, device, force):
    print("\n" + "=" * 70 + "\nSTAGE: VARIANTS (backbone × loss)\n" + "=" * 70)
    B.train_variants(cfg, manifest, device, force=force)


def run_research(device, manifest, force):
    """URD-v2 grid + SC-URD grid + B1 gate. Returns the loaded meta dict."""
    print("\n" + "=" * 70 + "\nSTAGE: RESEARCH (URD-v2 / SC-URD on frozen DINOv2)\n" + "=" * 70)
    dinov2, dino_tf = _load_dinov2(device)
    meta_cache = C.RESEARCH_DIR / f"urd_v2_meta_dinov2_embeddings_{TC.RESEARCH_CACHE_VERSION}.npz"
    meta = ME.load_or_extract_meta_embeddings(meta_cache, manifest, dinov2, dino_tf, device, force=force)

    # URD-v2: default + train-length ablations + cls-only.
    e_def = TC.URD_DEFAULT_EPOCHS
    epi = TC.EPISODES_PER_EPOCH
    R.train_urd_v2(meta, device, lambda_cons=0.5, suffix=f"full_e{e_def}",
                   epochs=e_def, episodes=epi, force=force)
    for ep in TC.URD_EPOCH_ABLATIONS:
        if ep != e_def:
            R.train_urd_v2(meta, device, lambda_cons=0.5, suffix=f"full_e{ep}",
                           epochs=ep, episodes=epi, force=force)
    R.train_urd_v2(meta, device, lambda_cons=0.0, suffix=f"cls_only_e{e_def}",
                   epochs=e_def, episodes=epi, force=force)

    # SC-URD residual β sweep.
    sc_def = TC.SC_URD_DEFAULT_EPOCHS
    for beta in TC.SC_URD_BETAS:
        if beta == "learnable":
            R.train_sc_urd(meta, device, beta=0.1, learnable_beta=True, suffix=f"scurd_rlearn_e{sc_def}",
                           epochs=sc_def, episodes=epi, force=force)
        else:
            suffix = f"scurd_r{str(beta).replace('.', '')}_e{sc_def}"
            R.train_sc_urd(meta, device, beta=float(beta), suffix=suffix,
                           epochs=sc_def, episodes=epi, force=force)

    # OOD-consistency variant (from the β=0.1 head).
    R.train_sc_urd(meta, device, beta=0.1, use_ood_cons=True,
                   suffix=f"scurd_r01_e{sc_def}_oodcons_fixed",
                   epochs=sc_def, episodes=epi, force=force)

    # Train-length ablations (e20 == SCURD_MAIN_CKPT).
    for ep in TC.SC_URD_TRAIN_EPOCH_ABLATIONS:
        suffix = f"scurd_r01_e{ep}"
        R.train_sc_urd(meta, device, beta=0.1, suffix=suffix, epochs=ep, episodes=epi, force=force)

    # Anchored-DINO heads.
    for beta in TC.SC_URD_DINO_ANCHOR_BETAS:
        suffix = f"scurd_anchor_b{str(beta).replace('.', '')}_e{sc_def}"
        R.train_sc_urd(meta, device, beta=beta, head_type="anchored_dino", suffix=suffix,
                       epochs=sc_def, episodes=epi, force=force)

    # Teacher-consistency heads.
    for eta in TC.SC_URD_DINO_TEACHER_ETAS:
        suffix = f"scurd_teacher_eta{str(eta).replace('.', '')}_e{sc_def}"
        R.train_sc_urd(meta, device, beta=0.1, teacher_cons_eta=eta, suffix=suffix,
                       epochs=sc_def, episodes=epi, force=force)

    # B1 learned density gate (needs trained backbones + DINOv2).
    try:
        from ..models import build_models
        models, transforms_out = build_models(device=device, need_clip=False, need_dinov2=True)
        b1_cache = C.RESEARCH_DIR / f"b1_metaval_expert_embeddings_{TC.RESEARCH_CACHE_VERSION}.npz"
        embs, labels = R.extract_b1_metaval_experts(manifest, models, transforms_out, device,
                                                    b1_cache, force=force)
        R.train_b1_gate(embs, labels, device, force=force)
    except Exception as e:
        print(f"  ⚠️ B1 gate skipped ({e}). Needs ArcFace-557 + ViT/SupCon variants + DINOv2.")
    return meta


def run_phase2(meta, device, force):
    print("\n" + "=" * 70 + "\nSTAGE: PHASE 2 (SC-URD cross-domain fine-tune)\n" + "=" * 70)
    ood_weak, ood_labels = ME.load_ood_dinov2()
    top50 = ME.select_top50_ood(ood_labels, n=50)
    epi = TC.EPISODES_PER_EPOCH
    configs = [
        ("scurd_r01_e10", "scurd_p2_8s8o_e10", 8, 8, 10, 0.3, 3e-4),
        ("scurd_r01_e10", "scurd_p2_8s8o_e20", 8, 8, 20, 0.3, 2e-4),
        ("scurd_r01_e10", "scurd_p2_4s12o_e10", 4, 12, 10, 0.3, 3e-4),
        ("scurd_r01_e10", "scurd_p2_8s8o_l01", 8, 8, 10, 0.1, 3e-4),
    ]
    for p1, p2, nws, nwo, ep, lam, lr in configs:
        try:
            R.train_sc_urd_phase2(meta, ood_weak, ood_labels, top50, device,
                                  phase1_suffix=p1, phase2_suffix=p2,
                                  n_way_swi=nws, n_way_ood=nwo, epochs=(2 if TC.FAST else ep),
                                  episodes=epi, lambda_cons=lam, lr=lr, force=force)
        except Exception as e:
            print(f"  ⚠️ Phase2 {p2} failed: {e}")


def run_scurd_seeds(device):
    print("\n" + "=" * 70 + "\nSTAGE: SC-URD seed checkpoints (validated engine)\n" + "=" * 70)
    from .._engines import variance_retrieval_evidence_colab as engine
    paths = engine.ensure_scurd_seed_checkpoints(device)
    print(f"  ✅ SC-URD seed checkpoints: {[p.name for p in paths]}")


def main():
    device = _device()
    cfg = TC.base_cfg()
    force = TC.FORCE_RETRAIN
    manifest = load_swi_manifest()

    need_images = TC.RUN_TRAIN_MAIN or TC.RUN_TRAIN_VARIANTS
    if need_images:
        print("\nPre-warming image cache (Drive → local) ...")
        preload_image_cache([p for p, _ in full_swi_items(manifest)])

    if TC.RUN_TRAIN_MAIN:
        run_main(cfg, manifest, device, force)
    if TC.RUN_TRAIN_VARIANTS:
        run_variants(cfg, manifest, device, force)

    meta = None
    if TC.RUN_TRAIN_RESEARCH:
        meta = run_research(device, manifest, force)
    if TC.RUN_TRAIN_PHASE2:
        if meta is None:
            dinov2, dino_tf = _load_dinov2(device)
            meta_cache = C.RESEARCH_DIR / f"urd_v2_meta_dinov2_embeddings_{TC.RESEARCH_CACHE_VERSION}.npz"
            meta = ME.load_or_extract_meta_embeddings(meta_cache, manifest, dinov2, dino_tf, device)
        run_phase2(meta, device, force)
    if TC.RUN_TRAIN_SCURD_SEEDS:
        run_scurd_seeds(device)

    print("\n✅ Training pipeline complete.")


if __name__ == "__main__":
    main()
