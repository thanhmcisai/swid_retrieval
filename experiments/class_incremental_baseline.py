# -*- coding: utf-8 -*-
"""Class-incremental CE baselines for the adaptation table.

These are stronger controls than naive CE fine-tuning: old classifier rows are
frozen, optional calibration uses old reference exemplars, and an iCaRL-style
nearest-exemplar-mean baseline evaluates whether a continual classifier with
exemplar memory converges toward the retrieval/prototype interface.
"""

import json
import os
import time

import numpy as np
import pandas as pd

from .. import config
from ..data import CachedImageLoader, canonical_label, get_transforms
from . import registry as R

RQ3_CIL_EPOCHS = int(os.environ.get("RQ3_CIL_EPOCHS", "30"))
RQ3_CIL_LR = float(os.environ.get("RQ3_CIL_LR", "1e-3"))
RQ3_CIL_BATCH_SIZE = int(os.environ.get("RQ3_CIL_BATCH_SIZE", "32"))
RQ3_CIL_OLD_CALIB_PER_SPECIES = int(os.environ.get("RQ3_CIL_OLD_CALIB_PER_SPECIES", "10"))


def _extract_features(model, paths, device, batch_size=64):
    import torch

    tfm = get_transforms(224, augment=False)
    loader = CachedImageLoader()
    feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            imgs = [tfm(image=loader.load(str(p)))["image"] for p in paths[i:i + batch_size]]
            x = torch.stack(imgs).to(device)
            feats.append(model.get_embedding(x, normalize=False).cpu())
    return torch.cat(feats, dim=0)


def run(ctx, out_dir, K=10, seed=42):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from ..models import _load_ce

    dev = ctx["device"]
    rng = np.random.RandomState(seed)
    model_ce, ce_ckpt = _load_ce(config.CKPT_CE_FULL, dev)
    ce_species = [canonical_label(s) for s in ce_ckpt.get("ce_species_list")]
    old_w = model_ce.classifier.weight.detach().cpu()
    old_b = model_ce.classifier.bias.detach().cpu()
    n_old, emb_dim = old_w.shape

    ood_df = pd.read_csv(config.OOD_IMAGES_CSV)
    splits, top50 = R.ood_species_splits()
    id_splits = R.id_species_splits()
    new_species = sorted(top50)

    ft_paths, ft_y = [], []
    for new_idx, sp in enumerate(new_species):
        pool = splits[sp]["pool_indices"]
        k_eff = min(K, len(pool))
        for j in rng.choice(len(pool), size=k_eff, replace=False):
            ft_paths.append(str(ood_df.iloc[pool[j]]["file_path"]))
            ft_y.append(n_old + new_idx)

    print(f"  class-incremental frozen-old CE baseline: {len(ft_paths)} refs, {len(new_species)} new species")
    t0 = time.time()
    x_train = _extract_features(model_ce, ft_paths, dev)
    y_train = torch.tensor(ft_y, dtype=torch.long)

    new_w = nn.Parameter(torch.empty(len(new_species), emb_dim))
    new_b = nn.Parameter(torch.zeros(len(new_species)))
    nn.init.xavier_uniform_(new_w)
    opt = torch.optim.AdamW([new_w, new_b], lr=RQ3_CIL_LR, weight_decay=1e-4)
    ds = TensorDataset(x_train, y_train)
    dl = DataLoader(ds, batch_size=RQ3_CIL_BATCH_SIZE, shuffle=True)

    old_w_dev = old_w.to(dev)
    old_b_dev = old_b.to(dev)
    for _ in range(RQ3_CIL_EPOCHS):
        for xb, yb in dl:
            xb = xb.to(dev)
            yb = yb.to(dev)
            logits_old = xb @ old_w_dev.T + old_b_dev
            logits_new = xb @ new_w.to(dev).T + new_b.to(dev)
            loss = F.cross_entropy(torch.cat([logits_old, logits_new], dim=1), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    elapsed = time.time() - t0
    all_species = ce_species + new_species
    new_w_cpu = new_w.detach().cpu()
    new_b_cpu = new_b.detach().cpu()

    sp_to_old_idx = {sp: i for i, sp in enumerate(ce_species)}
    new_to_idx = {sp: n_old + i for i, sp in enumerate(new_species)}

    old_query_paths, old_query_y = [], []
    old_calib_paths, old_calib_y = [], []
    for sp in sorted(id_splits):
        old_query_paths.extend(id_splits[sp]["query"])
        old_query_y.extend([sp_to_old_idx[sp]] * len(id_splits[sp]["query"]))
        old_pool = id_splits[sp]["pool"][:RQ3_CIL_OLD_CALIB_PER_SPECIES]
        old_calib_paths.extend(old_pool)
        old_calib_y.extend([sp_to_old_idx[sp]] * len(old_pool))

    new_query_paths, new_query_y = [], []
    for sp in new_species:
        q_paths = [str(ood_df.iloc[i]["file_path"]) for i in splits[sp]["query_indices"]]
        new_query_paths.extend(q_paths)
        new_query_y.extend([new_to_idx[sp]] * len(q_paths))

    x_old_query = _extract_features(model_ce, old_query_paths, dev).cpu()
    x_old_calib = _extract_features(model_ce, old_calib_paths, dev).cpu()
    x_new_query = _extract_features(model_ce, new_query_paths, dev).cpu()

    def _logits_from_feats(feats, gamma_new=0.0):
        logits_old = feats @ old_w.T + old_b
        logits_new = feats @ new_w_cpu.T + new_b_cpu - float(gamma_new)
        return torch.cat([logits_old, logits_new], dim=1)

    def _predict_feats(feats, gamma_new=0.0):
        pred = _logits_from_feats(feats, gamma_new=gamma_new).argmax(1).numpy()
        return np.asarray([all_species[int(i)] for i in pred])

    def _acc_by_species(paths_by_species, feats, gamma_new=0.0):
        pred = _predict_feats(feats, gamma_new=gamma_new)
        out, off = {}, 0
        for sp, n in paths_by_species:
            out[sp] = float((pred[off:off + n] == sp).mean()) if n else 0.0
            off += n
        return out

    old_blocks = [(sp, len(id_splits[sp]["query"])) for sp in sorted(id_splits)]
    new_blocks = [(sp, len(splits[sp]["query_indices"])) for sp in new_species]

    old_per = _acc_by_species(old_blocks, x_old_query, gamma_new=0.0)
    new_per = _acc_by_species(new_blocks, x_new_query, gamma_new=0.0)

    # Bias calibration uses only reference/calibration images: old public-ID pool
    # and the K-shot new-species references. It does not inspect the query sets.
    x_cal = torch.cat([x_old_calib, x_train.cpu()], dim=0)
    y_cal = np.asarray(old_calib_y + ft_y, dtype=int)
    gamma_grid = np.linspace(-30.0, 30.0, 241)
    best_gamma, best_score, best_old_cal, best_new_cal = 0.0, -1.0, 0.0, 0.0
    for gamma in gamma_grid:
        pred = _logits_from_feats(x_cal, gamma_new=float(gamma)).argmax(1).numpy()
        old_mask = y_cal < n_old
        new_mask = ~old_mask
        old_acc = float((pred[old_mask] == y_cal[old_mask]).mean()) if old_mask.any() else 0.0
        new_acc = float((pred[new_mask] == y_cal[new_mask]).mean()) if new_mask.any() else 0.0
        score = float(np.sqrt(max(old_acc, 0.0) * max(new_acc, 0.0)))
        if score > best_score:
            best_score, best_gamma, best_old_cal, best_new_cal = score, float(gamma), old_acc, new_acc

    old_per_cal = _acc_by_species(old_blocks, x_old_query, gamma_new=best_gamma)
    new_per_cal = _acc_by_species(new_blocks, x_new_query, gamma_new=best_gamma)

    def _proto_predict(q_feats, proto_mat, proto_labels):
        q = q_feats.numpy()
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        sims = q @ proto_mat.T
        return np.asarray(proto_labels)[sims.argmax(axis=1)]

    proto_vecs, proto_labels = [], []
    old_calib_y = np.asarray(old_calib_y)
    old_cal_np = x_old_calib.numpy()
    for sp in sorted(id_splits):
        idx = np.where(old_calib_y == sp_to_old_idx[sp])[0]
        if len(idx):
            proto_vecs.append(old_cal_np[idx].mean(axis=0))
            proto_labels.append(sp)
    x_train_np = x_train.numpy()
    ft_y_np = np.asarray(ft_y)
    for sp in new_species:
        idx = np.where(ft_y_np == new_to_idx[sp])[0]
        if len(idx):
            proto_vecs.append(x_train_np[idx].mean(axis=0))
            proto_labels.append(sp)
    proto_mat = np.asarray(proto_vecs, dtype=np.float32)
    proto_mat = proto_mat / np.maximum(np.linalg.norm(proto_mat, axis=1, keepdims=True), 1e-12)
    pred_old_proto = _proto_predict(x_old_query, proto_mat, proto_labels)
    pred_new_proto = _proto_predict(x_new_query, proto_mat, proto_labels)
    old_proto_per, new_proto_per = {}, {}
    off = 0
    for sp, n in old_blocks:
        old_proto_per[sp] = float((pred_old_proto[off:off + n] == sp).mean()) if n else 0.0
        off += n
    off = 0
    for sp, n in new_blocks:
        new_proto_per[sp] = float((pred_new_proto[off:off + n] == sp).mean()) if n else 0.0
        off += n

    variants = [
        {
            "strategy": "frozen_old_rows_new_rows_only",
            "acc_old_id_species": float(np.mean(list(old_per.values()))),
            "acc_new_species": float(np.mean(list(new_per.values()))),
            "updates_backbone": False,
            "updates_old_class_rows": False,
            "uses_rehearsal": False,
            "calibration_gamma_new": 0.0,
            "note": "Only new classifier rows are trained from K-shot OOD references; no old calibration/rehearsal is used.",
        },
        {
            "strategy": "frozen_old_rows_new_rows_bias_calibrated",
            "acc_old_id_species": float(np.mean(list(old_per_cal.values()))),
            "acc_new_species": float(np.mean(list(new_per_cal.values()))),
            "updates_backbone": False,
            "updates_old_class_rows": False,
            "uses_rehearsal": True,
            "old_calib_images_per_species": int(RQ3_CIL_OLD_CALIB_PER_SPECIES),
            "calibration_gamma_new": float(best_gamma),
            "calibration_old_acc": float(best_old_cal),
            "calibration_new_acc": float(best_new_cal),
            "note": "New-row logits are bias-calibrated on old reference-pool images and new K-shot references only.",
        },
        {
            "strategy": "icarl_nearest_exemplar_mean",
            "acc_old_id_species": float(np.mean(list(old_proto_per.values()))),
            "acc_new_species": float(np.mean(list(new_proto_per.values()))),
            "method_family": "iCaRL-style nearest-mean-of-exemplars",
            "updates_backbone": False,
            "updates_old_class_rows": False,
            "uses_rehearsal": True,
            "old_calib_images_per_species": int(RQ3_CIL_OLD_CALIB_PER_SPECIES),
            "new_exemplars_per_species": int(K),
            "calibration_gamma_new": None,
            "note": "iCaRL-style NME baseline on frozen CE features: old public-ID pool exemplars and new K-shot exemplars form class means; no representation retraining is performed.",
        },
    ]

    out = {
        "strategy": "frozen_old_rows_new_rows_only",
        "seed": int(seed),
        "K": int(K),
        "n_epochs": int(RQ3_CIL_EPOCHS),
        "lr": float(RQ3_CIL_LR),
        "updates_backbone": False,
        "updates_old_class_rows": False,
        "uses_rehearsal": False,
        "time_sec": float(elapsed),
        "acc_old_id_species": float(np.mean(list(old_per.values()))),
        "acc_new_species": float(np.mean(list(new_per.values()))),
        "old_per_species": old_per,
        "new_per_species": new_per,
        "variants": variants,
        "calibrated_old_per_species": old_per_cal,
        "calibrated_new_per_species": new_per_cal,
        "icarl_old_per_species": old_proto_per,
        "icarl_new_per_species": new_proto_per,
        "note": "Top-level metrics preserve the original new-rows-only baseline. See variants for bias-calibrated and iCaRL-style exemplar-mean controls.",
    }
    os.makedirs(str(out_dir), exist_ok=True)
    path = os.path.join(str(out_dir), "class_incremental_baseline.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    csv_path = os.path.join(str(out_dir), "class_incremental_baseline.csv")
    pd.DataFrame(variants).to_csv(csv_path, index=False)
    print(f"saved {path}: old={out['acc_old_id_species']:.3f} new={out['acc_new_species']:.3f}")
    print(f"saved {csv_path}")
    return out
