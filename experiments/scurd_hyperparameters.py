# -*- coding: utf-8 -*-
"""SC-URD hyperparameter-selection evidence on SWI meta-val only.

The selection protocol is intentionally separated from public-ID/OOD/VN26
reporting. It evaluates available SC-URD checkpoints and inference scoring
settings on a SmartWoodID meta-val support/query split, then writes the selected
configuration for reproducibility.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..scurd import load_scurd_model, project_np


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _softmax_np(x, axis=1):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.maximum(ex.sum(axis=axis, keepdims=True), 1e-12)


def _scurd_mode_config(mode):
    valid = {"raw", "centered", "logmeanexp", "prototype_mix",
             "centered_logmeanexp", "centered_prototype_mix"}
    if mode not in valid:
        raise ValueError(f"Invalid SC-URD memory mode: {mode!r}")
    return {
        "center": mode in {"centered", "centered_logmeanexp", "centered_prototype_mix"},
        "class_balance": mode in {"logmeanexp", "centered_logmeanexp"},
        "prototype_mix_alpha": 0.5 if mode in {"prototype_mix", "centered_prototype_mix"} else None,
    }


def _scurd_retrieval_eval(query_embs, query_labels, gallery_embs, gallery_labels,
                         top_m=50, tau=0.07, mode="centered"):
    cfg = _scurd_mode_config(mode)
    q = np.asarray(query_embs, dtype=np.float32)
    g = np.asarray(gallery_embs, dtype=np.float32)
    gl = np.asarray(gallery_labels)
    if cfg["center"]:
        mu = g.mean(axis=0, keepdims=True)
        q, g = q - mu, g - mu
    q, g = _norm(q), _norm(g)
    classes = np.asarray(sorted(set(gl.tolist())))
    c2i = {c: i for i, c in enumerate(classes)}
    sims = q @ g.T
    top_m_eff = min(int(top_m), sims.shape[1])
    top_idx = np.argpartition(-sims, top_m_eff - 1, axis=1)[:, :top_m_eff]
    top_sims = np.take_along_axis(sims, top_idx, axis=1)
    order = np.argsort(-top_sims, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_sims = np.take_along_axis(top_sims, order, axis=1)
    scores = np.full((len(q), len(classes)), -1e9, dtype=np.float64)
    tau = max(float(tau), 1e-6)
    for i in range(len(q)):
        top_labels = gl[top_idx[i]]
        for c in np.unique(top_labels):
            vals = top_sims[i, top_labels == c] / tau
            m = vals.max()
            score = m + np.log(np.exp(vals - m).sum())
            if cfg["class_balance"]:
                score -= np.log(max(1, np.sum(gl == c)))
            scores[i, c2i[c]] = score
    if cfg["prototype_mix_alpha"] is not None:
        proto = _norm(np.stack([g[gl == c].mean(axis=0) for c in classes]))
        proto_scores = (q @ proto.T) / tau
        scores = np.where(scores < -1e8, proto_scores, scores)
        a = float(cfg["prototype_mix_alpha"])
        scores = a * scores + (1.0 - a) * proto_scores
    probs = _softmax_np(scores, axis=1)
    preds = classes[np.argmax(probs, axis=1)]
    ql = np.asarray(query_labels)
    per_sp = {sp: float((preds[ql == sp] == sp).mean()) for sp in sorted(set(ql.tolist()))}
    return {"mean": float(np.mean(list(per_sp.values()))), "per_species": per_sp}


def _parse_csv_env(name, default, cast=str):
    values = []
    for item in str(os.environ.get(name, default)).split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    return values


def _load_meta_cache(research_dir):
    candidates = [
        Path(research_dir) / f"urd_v2_meta_dinov2_embeddings_{config.RESEARCH_CACHE_VERSION}.npz",
        config.RESEARCH_DIR / f"urd_v2_meta_dinov2_embeddings_{config.RESEARCH_CACHE_VERSION}.npz",
        config.ROOT_PATH / "results" / "paper_reframe" / "research_directions"
        / f"urd_v2_meta_dinov2_embeddings_{config.RESEARCH_CACHE_VERSION}.npz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        cache = np.load(path, allow_pickle=False)
        if {"val_weak", "val_labels"}.issubset(set(cache.files)):
            return path, {k: cache[k] for k in cache.files}
    raise FileNotFoundError(
        f"Missing urd_v2_meta_dinov2_embeddings_{config.RESEARCH_CACHE_VERSION}.npz "
        "with val_weak/val_labels in research_directions."
    )


def _meta_val_support_query(val_embs, val_labels, k_support=5, seed=42):
    labels = np.asarray([str(x) for x in val_labels])
    rng = np.random.RandomState(seed)
    support_idx, query_idx = [], []
    for sp in sorted(set(labels.tolist())):
        idx = np.where(labels == sp)[0]
        if len(idx) < 2:
            continue
        idx = idx.copy()
        rng.shuffle(idx)
        k = min(int(k_support), len(idx) - 1)
        support_idx.extend(idx[:k].tolist())
        query_idx.extend(idx[k:].tolist())
    if not support_idx or not query_idx:
        raise RuntimeError("Cannot build a non-empty meta-val support/query split.")
    support_idx = np.asarray(support_idx, dtype=int)
    query_idx = np.asarray(query_idx, dtype=int)
    return val_embs[support_idx], labels[support_idx], val_embs[query_idx], labels[query_idx]


def _discover_checkpoints(research_dir):
    research_dir = Path(research_dir)
    default = (
        "sc_urd_checkpoint_scurd_r01_e10_v2.pt,"
        "sc_urd_checkpoint_scurd_r01_e20_v2.pt,"
        "sc_urd_checkpoint_scurd_r01_e40_v2.pt,"
        "sc_urd_checkpoint_scurd_rlearn_e20_v2.pt,"
        "sc_urd_checkpoint_scurd_anchor_b005_e20_v2.pt,"
        "sc_urd_checkpoint_scurd_teacher_eta025_e20_v2.pt"
    )
    paths = []
    for pattern in _parse_csv_env("SCURD_HPARAM_CKPT_GLOBS", default):
        paths.extend(research_dir.glob(pattern))
    if not paths:
        paths = list(research_dir.glob(f"sc_urd_checkpoint_*_{config.SC_URD_CACHE_VERSION}.pt"))

    def key(path):
        return (0 if "scurd_r01_e20" in path.name else 1, path.name)

    return sorted(set(paths), key=key)


def run(run_root, force=False):
    """Run SC-URD hyperparameter selection and return the selected row."""
    run_root = Path(run_root)
    out_dir = run_root / "hyperparameters"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scurd_hyperparameter_selection.csv"
    json_path = out_dir / "scurd_hyperparameter_selection.json"
    force = force or os.environ.get("FORCE_HYPERPARAM_SELECTION", "0") == "1"
    if csv_path.exists() and json_path.exists() and not force:
        print(f"  ✅ SC-URD hyperparameter selection exists: {csv_path}")
        return json.load(open(json_path)).get("selected")

    research_dir = run_root / "research_directions"
    meta_path, meta = _load_meta_cache(research_dir)
    support_e, support_l, query_e, query_l = _meta_val_support_query(
        meta["val_weak"],
        meta["val_labels"],
        k_support=int(os.environ.get("SCURD_HPARAM_K_SUPPORT", "5")),
        seed=int(os.environ.get("SCURD_HPARAM_SPLIT_SEED", "42")),
    )
    modes = _parse_csv_env("SCURD_HPARAM_MODES", "raw,centered")
    taus = _parse_csv_env("SCURD_HPARAM_TAUS", str(config.SCURD_TAU), float)
    top_ms = _parse_csv_env("SCURD_HPARAM_TOP_MS", str(config.SCURD_TOP_M), int)
    ckpts = _discover_checkpoints(research_dir)
    if not ckpts:
        raise FileNotFoundError(f"No SC-URD checkpoints found in {research_dir}")

    import torch

    device = config.resolve_device()
    rows = []
    for ckpt_path in ckpts:
        try:
            model, ckpt = load_scurd_model(ckpt_path, in_dim=support_e.shape[1], device=device)
            support_h = project_np(model, support_e, device)
            query_h = project_np(model, query_e, device)
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "checkpoint": ckpt_path.name,
                "status": f"failed: {type(exc).__name__}: {exc}",
                "val_macro_r1": np.nan,
            })
            continue
        for mode in modes:
            for tau in taus:
                for top_m in top_ms:
                    try:
                        ev = _scurd_retrieval_eval(
                            query_h, query_l, support_h, support_l,
                            top_m=top_m, tau=tau, mode=mode)
                        val_macro_r1 = float(ev["mean"])
                        status = "ok"
                    except Exception as exc:  # noqa: BLE001
                        val_macro_r1 = np.nan
                        status = f"failed: {type(exc).__name__}: {exc}"
                    rows.append({
                        "checkpoint": ckpt_path.name,
                        "mode": mode,
                        "tau": float(tau),
                        "top_m": int(top_m),
                        "val_macro_r1": val_macro_r1,
                        "status": status,
                        "ckpt_epochs": ckpt.get("epochs"),
                        "ckpt_lambda_cons": ckpt.get("lambda_cons"),
                        "ckpt_beta": ckpt.get("beta"),
                        "ckpt_head_type": ckpt.get("head_type"),
                        "ckpt_learnable_beta": ckpt.get("learnable_beta"),
                    })
        del model
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    ok = df[df["status"].eq("ok") & df["val_macro_r1"].notna()].copy()
    if ok.empty:
        raise RuntimeError("No valid SC-URD hyperparameter-selection rows were produced.")
    ok["_default_tie"] = (
        ok["checkpoint"].astype(str).str.contains("scurd_r01_e20").astype(int)
        + ok["mode"].eq(config.SCURD_MODE).astype(int)
        + ok["tau"].eq(float(config.SCURD_TAU)).astype(int)
        + ok["top_m"].eq(int(config.SCURD_TOP_M)).astype(int)
    )
    ok = ok.sort_values(["val_macro_r1", "_default_tie"], ascending=[False, False])
    selected = ok.drop(columns=["_default_tie"]).iloc[0].to_dict()
    df["selected"] = (
        (df["checkpoint"] == selected["checkpoint"])
        & (df["mode"] == selected["mode"])
        & (df["tau"] == selected["tau"])
        & (df["top_m"] == selected["top_m"])
    )
    df.to_csv(csv_path, index=False)
    payload = {
        "selection_protocol": {
            "data": "SmartWoodID meta-val only",
            "meta_cache": str(meta_path),
            "support_images": int(len(support_l)),
            "query_images": int(len(query_l)),
            "species": int(len(set(np.asarray(meta["val_labels"]).astype(str).tolist()))),
            "k_support": int(os.environ.get("SCURD_HPARAM_K_SUPPORT", "5")),
            "split_seed": int(os.environ.get("SCURD_HPARAM_SPLIT_SEED", "42")),
            "modes": modes,
            "taus": taus,
            "top_ms": top_ms,
            "note": "Public-ID, OOD, and VN26 evaluation sets are not used for selection.",
        },
        "selected": selected,
        "rows": rows,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    print(f"  ✅ SC-URD hyperparameter selection selected: {selected}")
    print(f"  saved {csv_path}")
    return selected
