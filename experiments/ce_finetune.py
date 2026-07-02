# -*- coding: utf-8 -*-
"""Exp3.1 — CE fine-tune catastrophic forgetting (tab:adapt CE rows).

Lifted from final_metric_learning_cea_2026.py:3456-3573. Fine-tunes CE-Full on 50
new species (top-50 OOD), both freeze-backbone and full-finetune; reports old-ID
accuracy, new-species accuracy, forgetting, train time and peak memory. GPU + images
(gated by RUN_HEAVY in the orchestrator). Output → exp3_ce_finetune.json.

Note: training is not bit-reproducible across GPUs; seed is pinned to 42. Update
the paper's tab:adapt CE rows from the regenerated numbers.
"""

import copy
import json
import os
import time

import numpy as np
import pandas as pd

from .. import config
from ..data import canonical_label, CachedImageLoader, get_transforms
from . import registry as R

RQ3_CE_FT_EPOCHS = int(os.environ.get("RQ3_CE_FT_EPOCHS", "10"))


def _finetune_ce(strategy, ctx, K=10, seed=42):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from ..models import _load_ce

    dev = ctx["device"]
    model_ce, ce_ckpt = _load_ce(config.CKPT_CE_FULL, dev)
    ce_species = [canonical_label(s) for s in ce_ckpt.get("ce_species_list")]
    ood_df = pd.read_csv(config.OOD_IMAGES_CSV)
    splits, top50 = R.ood_species_splits()
    id_splits = R.id_species_splits()
    eval_tf = get_transforms(224, augment=False)
    aug_tf = get_transforms(224, augment=True)

    rng = np.random.RandomState(seed)
    new_species = sorted(top50)
    ft_paths, ft_labels = [], []
    for sp in new_species:
        pool = splits[sp]["pool_indices"]
        k_eff = min(K, len(pool))
        for j in rng.choice(len(pool), size=k_eff, replace=False):
            ft_paths.append(ood_df.iloc[pool[j]]["file_path"]); ft_labels.append(sp)

    model_ft = copy.deepcopy(model_ce).to(dev); model_ft.eval()
    old_cls = model_ft.classifier
    n_old, emb_dim = old_cls.out_features, old_cls.in_features
    new_cls = nn.Linear(emb_dim, n_old + len(new_species)).to(dev)
    with torch.no_grad():
        new_cls.weight[:n_old] = old_cls.weight
        new_cls.bias[:n_old] = old_cls.bias
    model_ft.classifier = new_cls
    all_species = ce_species + new_species
    sp_to_idx = {s: i for i, s in enumerate(all_species)}

    class _FT(Dataset):
        def __init__(self, paths, labels):
            self.paths, self.labels, self.loader = paths, labels, CachedImageLoader()
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            img = aug_tf(image=self.loader.load(self.paths[i]))["image"]
            return img, sp_to_idx[self.labels[i]]

    loader = DataLoader(_FT(ft_paths, ft_labels), batch_size=32, shuffle=True,
                        num_workers=config.num_workers(), pin_memory=True)
    if strategy == "freeze_backbone":
        for p in model_ft.backbone.parameters(): p.requires_grad_(False)
        for p in model_ft.head.parameters(): p.requires_grad_(False)
        model_ft.backbone.eval(); model_ft.head.eval()
        trainable = list(model_ft.classifier.parameters())
    else:
        for p in model_ft.parameters(): p.requires_grad_(True)
        trainable = list(model_ft.parameters())
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(RQ3_CE_FT_EPOCHS):
        model_ft.classifier.train() if strategy == "freeze_backbone" else model_ft.train()
        for imgs, lbls in loader:
            loss = crit(model_ft(imgs.to(dev)), lbls.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    model_ft.eval()

    loader_img = CachedImageLoader()

    def _predict(paths):
        preds = []
        with torch.no_grad():
            for i in range(0, len(paths), 64):
                imgs = [eval_tf(image=loader_img.load(p))["image"] for p in paths[i:i + 64]]
                lg = model_ft(torch.stack(imgs).to(dev))
                preds.extend([all_species[j] for j in lg.argmax(1).cpu().numpy()])
        return np.asarray(preds)

    new_per = {sp: float((_predict([ood_df.iloc[i]["file_path"] for i in splits[sp]["query_indices"]]) == sp).mean())
               for sp in new_species}
    old_per = {sp: float((_predict(id_splits[sp]["query"]) == sp).mean()) for sp in id_splits}
    del model_ft
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"strategy": strategy, "seed": seed, "K": K, "n_epochs": RQ3_CE_FT_EPOCHS,
            "time_sec": float(elapsed), "peak_memory_gb": float(peak),
            "acc_old_id_species": float(np.mean(list(old_per.values()))),
            "acc_new_species": float(np.mean(list(new_per.values()))),
            "old_per_species": old_per, "new_per_species": new_per}


def run(ctx, out_dir):
    """Writes exp3_ce_finetune.json (freeze + full strategies)."""
    out = {}
    for strat in ("freeze_backbone", "full_finetune"):
        print(f"  CE fine-tune ({strat})...")
        r = _finetune_ce(strat, ctx)
        out[strat] = r
        print(f"    old={r['acc_old_id_species']:.3f} new={r['acc_new_species']:.3f} "
              f"forgetting={r['acc_old_id_species']:.3f} time={r['time_sec']:.0f}s")
    os.makedirs(str(out_dir), exist_ok=True)
    path = os.path.join(str(out_dir), "exp3_ce_finetune.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"saved {path}")
    return out
