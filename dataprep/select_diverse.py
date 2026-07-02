# -*- coding: utf-8 -*-
"""Diverse-patch selection + image-level train/test directory split.

Lifted from old_version_claim_and_process_data/select_diverse_patches.py. Per
(image_id, patch_size) group of kept patches: extract ResNet50 features, KMeans
to MAX_PATCHES_PER_GROUP cluster-representative patches, then split image_ids
70/30 (seed 42) into train/ and test/ and copy survivors into FINAL_DATASET_DIR.
GPU-light (one ResNet50 forward pass over kept patches).
"""

import os
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from . import config as D


def _build_extractor(device):
    import torch
    from torchvision import models, transforms
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    extractor = torch.nn.Sequential(*list(model.children())[:-1]).eval().to(device)
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return extractor, preprocess


def get_features(image_paths, extractor, preprocess, device, batch_size=32):
    import torch
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Features", leave=False):
            tensors = []
            for p in image_paths[i:i + batch_size]:
                try:
                    tensors.append(preprocess(Image.open(p).convert("RGB")))
                except Exception:  # noqa: BLE001
                    continue
            if not tensors:
                continue
            out = extractor(torch.stack(tensors).to(device))
            feats.extend(out.squeeze().cpu().numpy())
    return np.array(feats)


def select_diverse_subset(paths, features, max_items):
    from sklearn.cluster import KMeans
    if len(paths) <= max_items:
        return paths
    km = KMeans(n_clusters=max_items, random_state=42, n_init="auto").fit(features)
    closest = np.argmin(km.transform(features), axis=0)
    return [paths[i] for i in closest]


def run():
    import torch
    metadata_path = D.PATCH_OUT_DIR / "metadata.csv"
    out_metadata = D.PATCH_OUT_DIR / "metadata_diverse_split.csv"
    if out_metadata.exists() and not D.FORCE_DATAPREP:
        print(f"✅ {out_metadata.name} exists → skip select_diverse")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"select_diverse on {device}")
    extractor, preprocess = _build_extractor(device)

    df = pd.read_csv(metadata_path)
    df_kept = df[df["status"] == "kept"].copy()
    final_kept = set()
    for _, group in tqdm(df_kept.groupby(["image_id", "patch_size"]), desc="Diverse groups"):
        paths = [os.path.join(str(D.PATCH_OUT_DIR), p) for p in group["relative_path"]]
        feats = get_features(paths, extractor, preprocess, device)
        if feats.shape[0] == 0:
            continue
        final_kept.update(select_diverse_subset(paths, feats, D.MAX_PATCHES_PER_GROUP))

    def diverse_status(row):
        if row["status"] != "kept":
            return row["status"]
        full = os.path.join(str(D.PATCH_OUT_DIR), row["relative_path"])
        return "kept_diverse" if full in final_kept else "removed_redundant"
    df["status"] = df.apply(diverse_status, axis=1)

    # Image-level 70/30 train/test split.
    from sklearn.model_selection import train_test_split
    df["split"] = ""
    diverse = df[df["status"] == "kept_diverse"]
    species = diverse["image_id"].unique()
    train_sp, test_sp = train_test_split(species, test_size=D.SELECT_TEST_SIZE, random_state=D.SEED)
    df.loc[df["image_id"].isin(train_sp), "split"] = "train"
    df.loc[df["image_id"].isin(test_sp), "split"] = "test"
    print(f"  train image_ids: {len(train_sp)} | test image_ids: {len(test_sp)}")

    out_metadata.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_metadata, index=False)
    print(f"💾 {out_metadata.name}: {df['status'].value_counts().to_dict()}")

    # Copy survivors into FINAL_DATASET_DIR/{train,test}/<relative_path>.
    kept_rows = df[df["status"] == "kept_diverse"]
    for _, row in tqdm(kept_rows.iterrows(), total=len(kept_rows), desc="Copy"):
        src = os.path.join(str(D.PATCH_OUT_DIR), row["relative_path"])
        dst = os.path.join(str(D.FINAL_DATASET_DIR), row["split"], row["relative_path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    print(f"✅ select_diverse done → {D.FINAL_DATASET_DIR}")


if __name__ == "__main__":
    run()
