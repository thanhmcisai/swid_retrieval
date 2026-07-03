# -*- coding: utf-8 -*-
"""Same-source OOD protocol.

The standard ID-vs-OOD split is taxonomic: public species overlapping
SmartWoodID are ID, all others are OOD. That is deployment-relevant, but it can
mix species novelty with acquisition-source differences. This experiment holds
the source dataset fixed: within each public source dataset, a subset of species
forms the known/reference set and a disjoint subset forms pseudo-OOD queries.

No raw images are read. The experiment consumes the public ID/OOD embedding
arrays already present in the full-954 cache and the expanded public CSV row
order used to create those arrays.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..data import canonical_label
from . import registry as R


def _infer_source(path):
    parts = Path(str(path)).parts
    if "datasets" in parts:
        i = parts.index("datasets")
        if i + 1 < len(parts):
            return parts[i + 1]
    text = str(path)
    marker = "/datasets/"
    if marker in text:
        rest = text.split(marker, 1)[1]
        return rest.split("/", 1)[0]
    return "unknown"


def _load_public_meta(labels_id, labels_ood):
    frames = []
    for kind, csv_path, labels in [
        ("id", config.ID_IMAGES_CSV, labels_id),
        ("ood", config.OOD_IMAGES_CSV, labels_ood),
    ]:
        df = pd.read_csv(csv_path).copy()
        if len(df) != len(labels):
            raise RuntimeError(
                f"{csv_path} has {len(df)} rows but embedding cache has {len(labels)} {kind} labels; "
                "expanded CSV row order must match the embedding cache."
            )
        df["cache_block"] = kind
        df["cache_index"] = np.arange(len(df), dtype=int)
        df["label"] = [canonical_label(x) for x in labels]
        df["source_dataset"] = df["file_path"].map(_infer_source)
        frames.append(df[["file_path", "label", "source_dataset", "cache_block", "cache_index"]])
    return pd.concat(frames, ignore_index=True)


def _method_public_embeddings(md):
    if md.get("ood") is None:
        return None
    return np.concatenate([md["id"], md["ood"]], axis=0)


def _source_protocol_indices(meta, source, seed=42, ood_fraction=0.3,
                             min_species=8, min_images_per_species=4,
                             refs_per_species=5):
    sub = meta[meta["source_dataset"].eq(source)].copy()
    counts = sub.groupby("label").size()
    species = sorted(counts[counts >= int(min_images_per_species)].index.tolist())
    if len(species) < int(min_species):
        return None
    source_offset = int(hashlib.md5(str(source).encode()).hexdigest()[:8], 16) % 100000
    rng = np.random.RandomState(int(seed) + source_offset)
    species = np.asarray(species, dtype=object)
    rng.shuffle(species)
    n_ood = max(2, int(round(len(species) * float(ood_fraction))))
    n_ood = min(n_ood, len(species) - 2)
    if n_ood < 1:
        return None
    ood_species = set(species[:n_ood].tolist())
    id_species = set(species[n_ood:].tolist())

    gallery_local, id_query_local, ood_query_local = [], [], []
    for sp in sorted(id_species):
        idx = sub.index[sub["label"].eq(sp)].to_numpy()
        idx = idx.copy()
        rng.shuffle(idx)
        k = min(int(refs_per_species), max(1, len(idx) // 2))
        gallery_local.extend(idx[:k].tolist())
        id_query_local.extend(idx[k:].tolist())
    for sp in sorted(ood_species):
        idx = sub.index[sub["label"].eq(sp)].to_numpy()
        ood_query_local.extend(idx.tolist())

    if not gallery_local or not id_query_local or not ood_query_local:
        return None
    return {
        "source_dataset": source,
        "id_species": sorted(id_species),
        "ood_species": sorted(ood_species),
        "gallery_local": np.asarray(gallery_local, dtype=int),
        "id_query_local": np.asarray(id_query_local, dtype=int),
        "ood_query_local": np.asarray(ood_query_local, dtype=int),
    }


def _to_public_pos(meta):
    """Map concatenated [ID embeddings, OOD embeddings] position for each meta row."""
    n_id = int(meta["cache_block"].eq("id").sum())
    block = meta["cache_block"].to_numpy()
    idx = meta["cache_index"].to_numpy(dtype=int)
    return np.where(block == "id", idx, n_id + idx)


def _scores(query, gallery):
    return 1.0 - (R._norm(query) @ R._norm(gallery).T).max(axis=1)


def _auroc(sid, sod):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sod))])
    return float(roc_auc_score(y, np.concatenate([sid, sod])))


def _fpr95(sid, sod):
    from sklearn.metrics import roc_curve
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sod))])
    fpr, tpr, _ = roc_curve(y, np.concatenate([sid, sod]))
    idx = np.searchsorted(tpr, 0.95)
    return float(fpr[min(idx, len(fpr) - 1)])


def _aupr(sid, sod):
    from sklearn.metrics import average_precision_score
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sod))])
    return float(average_precision_score(y, np.concatenate([sid, sod])))


def run(M, ctx, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_id, labels_ood = ctx["labels_id"], ctx["labels_ood"]
    meta = _load_public_meta(labels_id, labels_ood)
    public_pos = _to_public_pos(meta)

    min_species = int(os.environ.get("WITHIN_SOURCE_MIN_SPECIES", "8"))
    min_images = int(os.environ.get("WITHIN_SOURCE_MIN_IMAGES_PER_SPECIES", "4"))
    refs_per_species = int(os.environ.get("WITHIN_SOURCE_REFS_PER_SPECIES", "5"))
    ood_fraction = float(os.environ.get("WITHIN_SOURCE_OOD_FRACTION", "0.3"))
    seed = int(os.environ.get("WITHIN_SOURCE_SEED", "42"))
    source_filter = [
        s.strip() for s in os.environ.get("WITHIN_SOURCE_DATASETS", "").split(",")
        if s.strip()
    ]
    sources = source_filter or sorted(meta["source_dataset"].dropna().unique().tolist())

    protocols = []
    for src in sources:
        p = _source_protocol_indices(
            meta, src, seed=seed, ood_fraction=ood_fraction,
            min_species=min_species, min_images_per_species=min_images,
            refs_per_species=refs_per_species,
        )
        if p is not None:
            protocols.append(p)
    if not protocols:
        raise RuntimeError("No public source dataset satisfied the within-source OOD protocol requirements.")

    rows = []
    pooled = {}
    for method, md in M.items():
        emb_pub = _method_public_embeddings(md)
        if emb_pub is None:
            continue
        pooled_id, pooled_ood = [], []
        for p in protocols:
            g_pos = public_pos[p["gallery_local"]]
            id_pos = public_pos[p["id_query_local"]]
            od_pos = public_pos[p["ood_query_local"]]
            sid = _scores(emb_pub[id_pos], emb_pub[g_pos])
            sod = _scores(emb_pub[od_pos], emb_pub[g_pos])
            pooled_id.append(sid)
            pooled_ood.append(sod)
            rows.append({
                "method": method,
                "source_dataset": p["source_dataset"],
                "known_species": int(len(p["id_species"])),
                "heldout_ood_species": int(len(p["ood_species"])),
                "gallery_images": int(len(g_pos)),
                "id_query_images": int(len(id_pos)),
                "ood_query_images": int(len(od_pos)),
                "refs_per_species": int(refs_per_species),
                "ood_fraction": float(ood_fraction),
                "AUROC": _auroc(sid, sod),
                "FPR95": _fpr95(sid, sod),
                "AUPR": _aupr(sid, sod),
                "id_score_mean": float(np.mean(sid)),
                "ood_score_mean": float(np.mean(sod)),
                "separation_ood_minus_id": float(np.mean(sod) - np.mean(sid)),
            })
        if pooled_id:
            sid_all = np.concatenate(pooled_id)
            sod_all = np.concatenate(pooled_ood)
            pooled_row = {
                "AUROC": _auroc(sid_all, sod_all),
                "FPR95": _fpr95(sid_all, sod_all),
                "AUPR": _aupr(sid_all, sod_all),
                "id_images": int(len(sid_all)),
                "ood_images": int(len(sod_all)),
                "sources": int(len(protocols)),
            }
            pooled[method] = pooled_row
            rows.append({
                "method": method,
                "source_dataset": "POOLED",
                "known_species": None,
                "heldout_ood_species": None,
                "gallery_images": None,
                "id_query_images": pooled_row["id_images"],
                "ood_query_images": pooled_row["ood_images"],
                "refs_per_species": int(refs_per_species),
                "ood_fraction": float(ood_fraction),
                "AUROC": pooled_row["AUROC"],
                "FPR95": pooled_row["FPR95"],
                "AUPR": pooled_row["AUPR"],
                "id_score_mean": float(np.mean(sid_all)),
                "ood_score_mean": float(np.mean(sod_all)),
                "separation_ood_minus_id": float(np.mean(sod_all) - np.mean(sid_all)),
            })

    rows_csv = out_dir / "ood_within_source.csv"
    pd.DataFrame(rows).to_csv(rows_csv, index=False)
    payload = {
        "protocol": {
            "gallery_scope_for_model_registry": ctx.get("scope"),
            "source_datasets": [p["source_dataset"] for p in protocols],
            "min_species": min_species,
            "min_images_per_species": min_images,
            "refs_per_species": refs_per_species,
            "ood_fraction": ood_fraction,
            "seed": seed,
            "note": "Within each public source dataset, known-species gallery and held-out species queries are disjoint.",
        },
        "pooled": pooled,
        "per_source_rows": rows,
    }
    json_path = out_dir / "ood_within_source.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"saved {rows_csv} ({len(rows)} rows)")
    print(f"saved {json_path}")
    return payload
