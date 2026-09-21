# -*- coding: utf-8 -*-
"""Migrate public ID/OOD rows after label-correction without image extraction.

The full-954 cache contains two independent parts:
  1. SWI gallery embeddings, already correct and expensive to extract.
  2. Public ID/OOD query embeddings, whose row order follows
     ID/OOD_images_expanded.csv.

FSDM41 correction changes labels only; WoodAuth exclusion removes rows. This
module rewrites the public side by matching file_path rows from the previous
expanded CSV backups to the corrected expanded CSVs, then copies/reorders the
existing embedding/logit arrays and replaces labels. No images or models are
loaded.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

LOCAL_PREFIX = "/Users/admin/Downloads/Experimentals/datasets"


def _correction_file():
    candidates = [
        config.PUBLIC_LABEL_CORRECTIONS_JSON,
        config.ROOT_PATH / "dataset_label_corrections.json",
        Path(__file__).resolve().parents[2] / "dataset_label_corrections.json",
        Path.cwd() / "dataset_label_corrections.json",
    ]
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return Path(candidates[0])


def _sha256(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_path_series(s):
    datasets_dir = str(config.DATASETS_DIR)
    return s.astype(str).str.replace(LOCAL_PREFIX, datasets_dir, regex=False)


def _backup_for(path):
    path = Path(path)
    return path.with_name(path.stem + ".before_public_correction.csv")


def _load_old_new(side):
    if side == "id":
        new_path = config.ID_IMAGES_CSV
        backup = _backup_for(config.ID_IMAGES_CSV)
    elif side == "ood":
        new_path = config.OOD_IMAGES_CSV
        backup = _backup_for(config.OOD_IMAGES_CSV)
    else:
        raise ValueError(side)
    old_path = backup if backup.exists() else new_path
    if not old_path.exists():
        raise FileNotFoundError(f"Missing old expanded CSV for {side}: {old_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"Missing corrected expanded CSV for {side}: {new_path}")
    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)
    for p, df in [(old_path, old_df), (new_path, new_df)]:
        if "file_path" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{p} must contain file_path and label columns")
        df["file_path_norm"] = _norm_path_series(df["file_path"])
    return old_path, old_df, new_path, new_df


def _indexer_for_new_rows(old_df, new_df, side):
    if old_df["file_path_norm"].duplicated().any():
        dup = old_df.loc[old_df["file_path_norm"].duplicated(), "file_path_norm"].head().tolist()
        raise ValueError(f"Duplicate old {side} file_path rows, e.g. {dup}")
    old_index = {p: i for i, p in enumerate(old_df["file_path_norm"].tolist())}
    missing = [p for p in new_df["file_path_norm"].tolist() if p not in old_index]
    if missing:
        raise ValueError(
            f"{len(missing)} corrected {side} rows are not present in old cache CSV; "
            f"first missing: {missing[:3]}. Public cache migration cannot proceed without extraction."
        )
    return np.asarray([old_index[p] for p in new_df["file_path_norm"].tolist()], dtype=int)


def _migrate_side(src, out, side, old_df, new_df, indexer):
    prefixes = (f"embs_{side}_", f"logits_{side}_")
    label_prefix = f"labels_{side}_"
    n_old, n_new = len(old_df), len(new_df)
    label_values = new_df["label"].astype(str).to_numpy()
    for key in src.files:
        if key.startswith(prefixes):
            arr = src[key]
            if len(arr) != n_old:
                raise ValueError(f"{key} has {len(arr)} rows but old {side} CSV has {n_old}")
            out[key] = arr[indexer]
        elif key.startswith(label_prefix):
            arr = src[key]
            if len(arr) != n_old:
                raise ValueError(f"{key} has {len(arr)} rows but old {side} CSV has {n_old}")
            out[key] = np.asarray(label_values, dtype=str)
    return {
        "side": side,
        "old_rows": int(n_old),
        "new_rows": int(n_new),
        "dropped_rows": int(n_old - n_new),
        "species": int(new_df["label"].astype(str).nunique()),
    }


def run(force=None):
    force = config.FORCE_PUBLIC_CACHE_MIGRATION if force is None else bool(force)
    target = config.FULL954_CACHE_PATH
    meta_path = config.FULL954_META_PATH
    source = config.SOURCE_FULL954_CACHE_PATH
    if target.exists() and meta_path.exists() and not force:
        try:
            meta = json.load(open(meta_path))
            correction_path = _correction_file()
            current_hash = _sha256(correction_path) if Path(correction_path).exists() else None
            id_rows = len(pd.read_csv(config.ID_IMAGES_CSV)) if config.ID_IMAGES_CSV.exists() else None
            od_rows = len(pd.read_csv(config.OOD_IMAGES_CSV)) if config.OOD_IMAGES_CSV.exists() else None
            if (
                meta.get("public_correction_sha256") == current_hash
                and int(meta.get("id", {}).get("new_rows", -1)) == int(id_rows)
                and int(meta.get("ood", {}).get("new_rows", -1)) == int(od_rows)
            ):
                print(f"✅ Corrected public cache already valid: {target}")
                return target
            print("↻ Corrected public cache exists but metadata no longer matches current CSV/correction file; rebuilding.")
        except Exception as exc:  # noqa: BLE001
            print(f"↻ Could not validate corrected public cache metadata ({exc}); rebuilding.")
    if not source.exists():
        raise FileNotFoundError(
            f"Source full-954 cache not found: {source}. Set SOURCE_FULL954_CACHE_NAME "
            "to an existing uncorrected full cache, or build that cache before migration."
        )

    id_old_path, id_old, id_new_path, id_new = _load_old_new("id")
    od_old_path, od_old, od_new_path, od_new = _load_old_new("ood")
    id_indexer = _indexer_for_new_rows(id_old, id_new, "id")
    od_indexer = _indexer_for_new_rows(od_old, od_new, "ood")

    src = np.load(source, allow_pickle=False)
    out = {}
    for key in src.files:
        if key.startswith(("embs_id_", "labels_id_", "logits_id_", "embs_ood_", "labels_ood_", "logits_ood_")):
            continue
        out[key] = src[key]
    id_summary = _migrate_side(src, out, "id", id_old, id_new, id_indexer)
    od_summary = _migrate_side(src, out, "ood", od_old, od_new, od_indexer)

    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **out)
    tmp.replace(target)

    labels_ood = set(id_new["label"].astype(str)) & set(od_new["label"].astype(str))
    if labels_ood:
        raise RuntimeError(f"Corrected ID/OOD overlap detected: {sorted(labels_ood)[:5]}")
    correction_path = _correction_file()
    meta = {
        "artifact": "embedding_cache_full954_corrected_public",
        "source_cache": str(source),
        "target_cache": str(target),
        "swi_full_species": int(len(set(str(x) for x in out.get("labels_swi_dinov2", [])))),
        "public_correction_file": str(correction_path),
        "public_correction_sha256": _sha256(correction_path) if Path(correction_path).exists() else None,
        "id_csv": str(id_new_path),
        "ood_csv": str(od_new_path),
        "old_id_csv": str(id_old_path),
        "old_ood_csv": str(od_old_path),
        "id": id_summary,
        "ood": od_summary,
        "notes": [
            "SWI gallery arrays copied unchanged from source cache.",
            "Public arrays migrated by file_path; labels replaced from corrected expanded CSVs.",
            "WoodAuth rows are absent if excluded during public dataprep.",
        ],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"✅ Wrote corrected public cache: {target}")
    print(f"✅ Wrote corrected public cache meta: {meta_path}")
    print(f"  ID:  {id_summary}")
    print(f"  OOD: {od_summary}")
    return target


def main():
    return run()


if __name__ == "__main__":
    main()
