# -*- coding: utf-8 -*-
"""SC-URD backbone ablation — the method search.

SC-URD currently adapts DINOv2 (the weakest single embedding). This runs the SAME
SC-URD recipe (residual adapter + gallery-conditioned centering) on stronger bases
{DINOv2, ArcFace-557, Fusion} and compares them on the axes that decide the proposed
method:
  - matched-954 Prototype + R@1 (the fair paradigm comparison — the number that was the problem)
  - deployment-24 R@1 (centered scoring)
  - OOD AUROC / FPR95 (deployment-24)
Picks the base with the best balanced score as "SC-URD (proposed)". Output:
  scurd_backbone_ablation.{csv,json}.

HEAVY + opt-in (RUN_SCURD_BACKBONE=1): for arcface557/fusion it extracts base meta
embeddings (image pass) and trains the adapter; on Colab with the SWI images. Each
base is guarded so a partial failure does not abort the rest.
"""

import gc
import json
import os

import numpy as np

from .. import config
from ..data import canonical_label
from ..gallery import build_gallery_mask
from . import registry as R
from .ood_baselines import _auroc, _fpr95

CHUNK = int(os.environ.get("SCURD_BACKBONE_CHUNK", "256"))


def _r1_chunked(q, ql, g, gl, mode):
    """Public-ID R@1 (given scoring mode), chunked over query rows — memory-safe at 954."""
    preds = []
    for i in range(0, len(q), CHUNK):
        scores, classes = R.scurd_class_scores(q[i:i + CHUNK], g, gl, mode=mode)
        preds.append(np.asarray(classes)[scores.argmax(axis=1)])
    preds = np.concatenate(preds) if preds else np.array([])
    return float(R._macro_from_preds(preds, ql)[0])


def _cosdist_chunked(q, gn):
    """1 - max cosine sim to gallery, chunked. gn = pre-L2-normalized gallery."""
    out = np.empty(len(q), dtype=np.float64)
    for i in range(0, len(q), CHUNK):
        out[i:i + CHUNK] = 1.0 - (R._norm(q[i:i + CHUNK]) @ gn.T).max(axis=1)
    return out

# base → (raw id/swi/ood emb keys or fusion), meta-cache name, adapter suffix, in_dim.
BASE_SPECS = {
    "dinov2": {"id": "embs_id_dinov2", "swi": "embs_swi_dinov2", "ood": "embs_ood_dinov2",
               "meta": "urd_v2_meta_dinov2_embeddings_v2.npz", "suffix": "scurd_r01_e20", "in_dim": 768},
    "arcface557": {"id": "embs_id_arc", "swi": "embs_swi_arc", "ood": "embs_ood_arc",
                   "meta": "scurd_meta_arcface557_v2.npz", "suffix": "scurd_arcface557_r01_e20", "in_dim": 512},
    "fusion": {"fuse": True, "meta": "scurd_meta_fusion_v2.npz",
               "suffix": "scurd_fusion_r01_e20", "in_dim": 1280},
}


def _raw(spec, emb, which):
    """Raw (pre-adapter) embeddings for id/swi/ood of a base."""
    if spec.get("fuse"):
        a = {"id": "embs_id_arc", "swi": "embs_swi_arc", "ood": "embs_ood_arc"}[which]
        d = {"id": "embs_id_dinov2", "swi": "embs_swi_dinov2", "ood": "embs_ood_dinov2"}[which]
        return R._fuse(emb[a], emb[d])
    return emb[spec[which]]


def _ensure_adapter(base, spec, emb, device):
    """Return a loaded SC-URD adapter for `base`, training it if the ckpt is absent."""
    from ..scurd import load_scurd_model
    ckpt = config.RESEARCH_DIR / f"sc_urd_checkpoint_{spec['suffix']}_{config.SC_URD_CACHE_VERSION}.pt"
    if not ckpt.exists():
        # Train: extract this base's meta embeddings, then train the adapter on the
        # cached vectors. Load ONLY the backbone(s) this base needs (not all of
        # build_models) to avoid GPU OOM, and free them before adapter training.
        from ..training import meta_embeddings as ME
        from ..training.train_research import train_sc_urd
        from ..training import config as TC
        from ..data import load_swi_manifest
        meta_path = config.RESEARCH_DIR / spec["meta"]
        manifest = load_swi_manifest()
        dino_cache = config.RESEARCH_DIR / "urd_v2_meta_dinov2_embeddings_v2.npz"
        if meta_path.exists():
            meta = np_load_meta(meta_path)              # same key schema for any base
        elif base == "dinov2":
            dinov2, dtf = _load_dinov2(device)
            meta = ME.load_or_extract_meta_embeddings(meta_path, manifest, dinov2, dtf, device)
            _free(dinov2, device)
        else:
            # arcface557 + fusion: only the ArcFace ConvNeXt is needed (fusion reuses the
            # DINOv2 meta cache); load DINOv2 ONLY as a fallback if that cache is absent.
            from ..models import _load_metric
            models = {"arc": _load_metric(config.CKPT_ARC_557, "convnext_base", 512, device)[0]}
            tfs = {}
            if base == "fusion" and not dino_cache.exists():
                dino, dtf = _load_dinov2(device)
                models["dinov2"] = dino; tfs["dinov2"] = dtf
            meta = ME.extract_base_meta(base, meta_path, manifest, models, tfs, device, dino_cache=dino_cache)
            for m in list(models.values()):
                _free(m, device)
        epochs = TC.SC_URD_DEFAULT_EPOCHS if base == "dinov2" else 20
        train_sc_urd(meta, device, beta=0.1, suffix=spec["suffix"],
                     epochs=epochs, episodes=TC.EPISODES_PER_EPOCH)
    model, _ = load_scurd_model(ckpt, in_dim=spec["in_dim"], device=device)
    return model


def np_load_meta(path):
    c = np.load(path, allow_pickle=False)
    return {k: c[k] for k in c.files}


def _free(model, device):
    try:
        import torch
        del model
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _load_dinov2(device):
    import torch
    from torchvision import transforms
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to(device)
    tf = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return m, tf


def _eval_base(base, spec, emb, labels_id, mask_d, gl_d, mask_f, gl_f, test_mask, device):
    from ..scurd import project_np
    model = _ensure_adapter(base, spec, emb, device)
    pid = project_np(model, _raw(spec, emb, "id"), device)
    pood = project_np(model, _raw(spec, emb, "ood"), device)
    pswi = project_np(model, _raw(spec, emb, "swi"), device)   # 176k × 512 (projected)

    row = {"base": base, "adapter_in_dim": spec["in_dim"]}
    # matched-954 (full gallery): prototype uses 954 centroids (cheap); R@1 is chunked.
    g_f = pswi[mask_f]
    row["matched954_prototype"] = float(R.prototype_classify_per_species(pid, labels_id, g_f, gl_f)[0])
    row["matched954_R1"] = _r1_chunked(pid, labels_id, g_f, gl_f, "centered")
    # deployment-24
    g_d = pswi[mask_d]
    row["deploy24_R1"] = _r1_chunked(pid, labels_id, g_d, gl_d, "centered")
    # OOD (deployment-24 gallery)
    gn_d = R._norm(g_d)
    sid = _cosdist_chunked(pid, gn_d); sod = _cosdist_chunked(pood, gn_d)[test_mask]
    row["ood_auroc"] = float(_auroc(sid, sod))
    row["ood_fpr95"] = float(_fpr95(sid, sod))
    del pid, pood, pswi, g_f, g_d, gn_d
    gc.collect()
    return row


def run(M, ctx, out_dir):
    """Compare the SC-URD recipe across backbones. M/ctx from build_M (id_only).

    Memory-safe: gallery masks are derived directly (no build_M at full_swi, which
    would materialize every method's 176k gallery), and the 954-gallery R@1/OOD
    scoring is chunked over query rows.
    """
    emb = ctx["emb"]
    device = config.resolve_device()
    labels_id = ctx["labels_id"]
    labels_swi = ctx.get("labels_swi")
    if labels_swi is None:
        labels_swi = np.array([canonical_label(x) for x in emb["labels_swi_dinov2"]])
    ce_mask = emb["swi_in_ce_train"] if "swi_in_ce_train" in emb.files else None
    test_mask = R.ood_test_mask(ctx["labels_ood"])
    mask_d, gl_d, _ = build_gallery_mask(labels_swi, set(labels_id.tolist()), scope="id_only", ce_train_mask=ce_mask)
    mask_f, gl_f, _ = build_gallery_mask(labels_swi, set(labels_id.tolist()), scope="full_swi", ce_train_mask=ce_mask)

    rows = []
    for base, spec in BASE_SPECS.items():
        try:
            rows.append(_eval_base(base, spec, emb, labels_id, mask_d, gl_d, mask_f, gl_f, test_mask, device))
            print(f"  SC-URD/{base}: matched954 proto={rows[-1]['matched954_prototype']:.3f} "
                  f"deploy24 R@1={rows[-1]['deploy24_R1']:.3f} OOD={rows[-1]['ood_auroc']:.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"  SC-URD/{base} failed ({type(e).__name__}: {e}); skipping.")
            gc.collect()

    best = None
    if rows:
        def _score(r):  # balanced: matched-954 fairness + deployment + OOD
            return r["matched954_prototype"] + r["deploy24_R1"] + r["ood_auroc"]
        best = max(rows, key=_score)["base"]
    out = {"rows": rows, "proposed_base": best,
           "score": "matched954_prototype + deploy24_R1 + ood_auroc"}
    os.makedirs(str(out_dir), exist_ok=True)
    with open(os.path.join(str(out_dir), "scurd_backbone_ablation.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=float)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(str(out_dir), "scurd_backbone_ablation.csv"), index=False)
    except Exception:  # noqa: BLE001
        pass
    print(f"  ✅ scurd_backbone: {len(rows)} bases; proposed = {best}")
    return out


if __name__ == "__main__":
    M, ctx = R.build_M("id_only", with_exp4=False, with_scurd=False)
    run(M, ctx, config.RESULTS_DIR / "scurd_backbone_standalone")
