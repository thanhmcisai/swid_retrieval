# -*- coding: utf-8 -*-
"""Class-incremental CE baseline with frozen old-class rows.

This is a stronger control than naive CE fine-tuning for the adaptation table:
the CE-Full backbone/head and old classifier rows are frozen, and only 50 new
classifier rows are trained from K=10 OOD references. It tests whether the
old=0 collapse in the naive fine-tune rows is an intrinsic classifier limit or a
no-rehearsal/trainable-old-row artifact.
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

    def _predict(paths):
        feats = _extract_features(model_ce, paths, dev).cpu()
        logits_old = feats @ old_w.T + old_b
        logits_new = feats @ new_w_cpu.T + new_b_cpu
        pred = torch.cat([logits_old, logits_new], dim=1).argmax(1).numpy()
        return np.asarray([all_species[int(i)] for i in pred])

    old_per = {sp: float((_predict(id_splits[sp]["query"]) == sp).mean()) for sp in id_splits}
    new_per = {}
    for sp in new_species:
        q_paths = [str(ood_df.iloc[i]["file_path"]) for i in splits[sp]["query_indices"]]
        new_per[sp] = float((_predict(q_paths) == sp).mean())

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
        "note": "Frozen CE-Full backbone/head and frozen old classifier rows; only new rows are trained from K-shot OOD references.",
    }
    os.makedirs(str(out_dir), exist_ok=True)
    path = os.path.join(str(out_dir), "class_incremental_baseline.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"saved {path}: old={out['acc_old_id_species']:.3f} new={out['acc_new_species']:.3f}")
    return out
