"""Re-evaluate native CE from images, without training or historical fallback.

The fresh query extraction avoids binding a new classifier map to legacy logits.
Outputs are audit artifacts, not an automatic replacement of the full cache.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--id-csv", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    for path in (args.checkpoint, args.id_csv, args.cache):
        if path is None:
            continue
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    if args.out.exists():
        parser.error("Output already exists; choose a fresh audit directory")
    os.environ["ROOT_PATH"] = str(args.root)

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from ..data import CSVImageDataset, get_transforms
    from ..embeddings.extract import extract_embeddings
    from ..models import CEClassifier
    from .rq1_native import native_ce_macro

    frame = pd.read_csv(args.id_csv)
    if "file_path" not in frame or frame["file_path"].duplicated().any():
        raise ValueError("ID CSV must contain unique file_path rows")
    missing = [p for p in frame["file_path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} query images missing; first: {missing[:3]}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = ckpt.get("ce_species_list")
    if classes is None:
        raise ValueError("Checkpoint has no ordered ce_species_list; do not guess it")
    provenance = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": digest(args.checkpoint),
                  "id_csv": str(args.id_csv), "id_csv_sha256": digest(args.id_csv),
                  "epoch": ckpt.get("epoch"), "n_queries": len(frame),
                  "n_classes": len(classes), "transform": "get_transforms(224, augment=False)",
                  "training_performed": False, "legacy_results_used": False}
    print(json.dumps(provenance, indent=2))
    if args.preflight:
        return
    model = CEClassifier("convnext_base", n_classes=len(classes), embedding_dim=512, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(args.device).eval()
    ds = CSVImageDataset(frame, transform=get_transforms(224, augment=False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    features, label_indices, logits = extract_embeddings(model, loader, args.device, return_logits=True)
    labels = np.asarray([ds.idx_to_class[int(i)] for i in label_indices], dtype=str)
    if not np.array_equal(labels, frame["label"].to_numpy(dtype=str)):
        raise ValueError("Extracted query order differs from the input CSV")
    if args.cache is not None:
        with np.load(args.cache, allow_pickle=False) as cache:
            for key, fresh in (("embs_id_ce_full_norm", features),
                               ("logits_id_ce_full", logits)):
                cached = cache[key]
                if cached.shape != fresh.shape:
                    raise ValueError(f"{key}: cached shape differs from fresh extraction")
                diff = float(np.max(np.abs(cached.astype(np.float32) - fresh.astype(np.float32))))
                provenance[key + "_max_abs_diff"] = diff
                if diff > 1e-3:
                    raise ValueError(f"{key}: fresh extraction differs from v5 cache (max_abs_diff={diff})")
            if "labels_id_dinov2" in cache and not np.array_equal(
                    np.char.lower(np.char.replace(cache["labels_id_dinov2"].astype(str), " ", "_")),
                    np.char.lower(np.char.replace(labels.astype(str), " ", "_"))):
                raise ValueError("v5 cache ID labels differ from fresh CE query order")
        provenance["cache_sha256"] = digest(args.cache)
    result = native_ce_macro(logits, classes, labels)
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out / "native_ce_queries.npz", logits_id_ce_full=logits,
                        embs_id_ce_full_norm=features,
                        labels_id=labels, labels_id_ce_full=labels,
                        paths_id=frame["file_path"].to_numpy(dtype=str),
                        paths_id_ce_full=frame["file_path"].to_numpy(dtype=str),
                        ce_species_list=np.asarray(classes, dtype=str),
                        ce_full_checkpoint_sha256=np.asarray(provenance["checkpoint_sha256"]))
    (args.out / "native_ce_audit.json").write_text(json.dumps({**provenance, **result}, indent=2) + "\n")
    print(f"Native CE macro accuracy: {result['mean']:.9f}; saved to {args.out}")


if __name__ == "__main__":
    main()
