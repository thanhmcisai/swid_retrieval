# -*- coding: utf-8 -*-
"""Anti-bug diagnostics for the full-954 RQ1 collapse.

Reads the already-built embedding_cache_full954_v3.npz (no extraction) and runs
three checks:

  1. DECISIVE reproduction: under GALLERY_SCOPE=id_only the new cache must
     reproduce the published 24-gallery numbers (DINOv2 R@1~0.357, ArcFace~0.238,
     CE-Full~0.306, Fusion~0.348, SC-URD~0.579). If it does, the embeddings / NN /
     prototype / SC-URD projection are correct, so the low full_swi numbers are a
     genuine cardinality effect, not a bug.
  2. R@1/5/20/100 curves per scope: if the correct species is reachable deep in
     the ranking (R@100 high) but not at top-1, that is the expected effect of
     ~954 distractors, not broken embeddings.
  3. SC-URD raw vs centered: detect whether gallery-mean centering at 176k-image
     scale is hurting SC-URD specifically.
  4. Same-domain leave-one-out (SWI->SWI) R@1 over the full gallery: should be
     high; if same-domain is high but cross-domain (public->SWI) is low, the gap
     is domain shift, not a bug.

    python -m swid_retrieval.diagnose
"""

import numpy as np

from . import config
from .data import canonical_label
from .experiments.rq1_native import prototype_macro_top1

OLD_IDONLY_R1 = {"DINOv2": 0.357, "ArcFace-557": 0.238, "CE-Full": 0.306,
                 "Fusion": 0.348, "SC-URD": 0.579}


def _norm(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _fuse(a, b):
    return _norm(np.concatenate([_norm(a), _norm(b)], axis=1))


def recall_at_k_macro(q, ql, g, gl, ks=(1, 5, 20, 100), centered=False, batch=256):
    """Macro-per-query-species Recall@k (hit if any same-species in top-k)."""
    q = _norm(q); g = _norm(g)
    if centered:
        mu = g.mean(0, keepdims=True)
        q = _norm(q - mu); g = _norm(g - mu)
    gl = np.asarray(gl); ql = np.asarray(ql)
    ks = [k for k in ks if k < len(gl)] or [1]
    kmax = max(ks)
    corr = {k: np.zeros(len(ql), bool) for k in ks}
    for i in range(0, len(q), batch):
        s = q[i:i + batch] @ g.T
        part = np.argpartition(-s, kmax - 1, axis=1)[:, :kmax]
        rows = np.arange(part.shape[0])[:, None]
        order = part[rows, np.argsort(-s[rows, part], axis=1)]
        b = order.shape[0]
        for k in ks:
            tk = order[:, :k]
            corr[k][i:i + b] = (gl[tk] == ql[i:i + b][:, None]).any(1)
    return {k: float(np.mean([corr[k][ql == c].mean() for c in np.unique(ql)])) for k in ks}


def same_domain_r1(g_emb, g_labels, n_query=2000, batch=256, seed=0):
    """Leave-one-out SWI->SWI R@1 over the full gallery (self excluded)."""
    g = _norm(g_emb); gl = np.asarray(g_labels)
    rng = np.random.RandomState(seed)
    qi = rng.choice(len(g), min(n_query, len(g)), replace=False)
    hits = 0
    for i in range(0, len(qi), batch):
        idx = qi[i:i + batch]
        s = g[idx] @ g.T
        s[np.arange(len(idx)), idx] = -1e9   # exclude self
        nn = s.argmax(1)
        hits += int((gl[nn] == gl[idx]).sum())
    return hits / len(qi)


def main():
    print(f"cache = {config.FULL954_CACHE_PATH}")
    emb = np.load(config.FULL954_CACHE_PATH, allow_pickle=False)
    labels_id = np.array([canonical_label(x) for x in emb["labels_id_dinov2"]])
    labels_swi = np.array([canonical_label(x) for x in emb["labels_swi_dinov2"]])
    id_species = set(labels_id)

    scopes = {
        "id_only": np.array([x in id_species for x in labels_swi]),
        "full_swi": np.ones(len(labels_swi), bool),
    }
    if "swi_in_ce_train" in emb.files:
        scopes["ce_train"] = np.asarray(emb["swi_in_ce_train"], bool)

    base = {
        "DINOv2": (emb["embs_id_dinov2"], emb["embs_swi_dinov2"]),
        "ArcFace-557": (emb["embs_id_arc"], emb["embs_swi_arc"]),
        "CE-Full": (emb["embs_id_ce_full_norm"], emb["embs_swi_ce_full_norm"]),
        "Fusion": (_fuse(emb["embs_id_arc"], emb["embs_id_dinov2"]),
                   _fuse(emb["embs_swi_arc"], emb["embs_swi_dinov2"])),
    }

    sc_id = sc_swi = None
    if config.SCURD_MAIN_CKPT.exists():
        from .scurd import load_scurd_model, project_np
        dev = config.resolve_device()
        print(f"projecting SC-URD from {config.SCURD_MAIN_CKPT.name} on {dev}...")
        model, _ = load_scurd_model(config.SCURD_MAIN_CKPT,
                                    in_dim=emb["embs_id_dinov2"].shape[1], device=dev)
        sc_id = project_np(model, emb["embs_id_dinov2"], dev)
        sc_swi = project_np(model, emb["embs_swi_dinov2"], dev)
    else:
        print(f"WARNING: SC-URD ckpt missing ({config.SCURD_MAIN_CKPT}); skipping SC-URD rows.")

    hdr = f"{'method':14s} {'proto':>8} {'R@1':>8} {'R@5':>8} {'R@20':>8} {'R@100':>8}   ref(old id_only R@1)"
    for scope, mask in scopes.items():
        gl = labels_swi[mask]
        print(f"\n=== scope={scope}: {int(mask.sum())} imgs, {len(set(gl))} species ===")
        print(hdr)
        for name, (qe, ge) in base.items():
            g = ge[mask]
            proto = prototype_macro_top1(qe, labels_id, g, gl)["mean"]
            rk = recall_at_k_macro(qe, labels_id, g, gl)
            ref = f"   ~{OLD_IDONLY_R1[name]:.3f}" if scope == "id_only" else ""
            print(f"{name:14s} {proto:8.4f} {rk.get(1,float('nan')):8.4f} {rk.get(5,float('nan')):8.4f} "
                  f"{rk.get(20,float('nan')):8.4f} {rk.get(100,float('nan')):8.4f}{ref}")
        if sc_id is not None:
            g = sc_swi[mask]
            for tag, cen in [("SC-URD(raw)", False), ("SC-URD(cen)", True)]:
                proto = prototype_macro_top1(sc_id, labels_id, g, gl, centered=cen)["mean"]
                rk = recall_at_k_macro(sc_id, labels_id, g, gl, centered=cen)
                ref = f"   ~{OLD_IDONLY_R1['SC-URD']:.3f}" if scope == "id_only" and cen else ""
                print(f"{tag:14s} {proto:8.4f} {rk.get(1,float('nan')):8.4f} {rk.get(5,float('nan')):8.4f} "
                      f"{rk.get(20,float('nan')):8.4f} {rk.get(100,float('nan')):8.4f}{ref}")

    print("\n=== same-domain sanity: SWI->SWI leave-one-out R@1 over the FULL 954 gallery ===")
    for name, (_qe, ge) in base.items():
        r1 = same_domain_r1(ge, labels_swi)
        print(f"  {name:14s} SWI->SWI R@1 = {r1:.4f}   (expect high if embeddings are healthy)")

    print("\nINTERPRETATION:")
    print("  - id_only rows ~= the 'ref' column  -> pipeline correct; full_swi drop is real cardinality.")
    print("  - R@100 >> R@1 under full_swi        -> correct species is findable; top-1 hard due to 954 distractors.")
    print("  - SC-URD(raw) >> SC-URD(cen)          -> gallery-mean centering hurts at scale; re-run engine with SCURD_MAIN_MODE=raw.")
    print("  - SWI->SWI R@1 high but public->SWI low -> the gap is domain shift, not a bug.")


if __name__ == "__main__":
    main()
