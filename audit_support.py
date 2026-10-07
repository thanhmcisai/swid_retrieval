"""Small, dependency-light provenance and bounded-memory evaluation helpers."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def canonical(labels):
    return np.asarray([str(x).strip().lower().replace(" ", "_") for x in labels])


def check_labels(expected, actual, context):
    if not np.array_equal(canonical(expected), canonical(actual)):
        raise ValueError(f"Ordered label mismatch: {context}; equal row counts are insufficient")


def top_matches(q, g, k=1, batch=64):
    """Exact global top-k on already-normalized rows; bounded query dimension."""
    if not len(q) or not len(g) or k < 1 or batch < 1:
        raise ValueError("Nonempty features and positive k/batch required")
    k = min(k, len(g))
    indices, scores = [], []
    for start in range(0, len(q), batch):
        sims = q[start:start + batch] @ g.T
        if k == 1:
            idx = sims.argmax(axis=1)[:, None]
        else:
            idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        indices.append(np.take_along_axis(idx, order, axis=1))
        scores.append(np.take_along_axis(vals, order, axis=1))
    return np.concatenate(indices), np.concatenate(scores)


def completed(out, stage):
    path = Path(out) / (stage + ".done.json")
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    for name, digest in record["outputs"].items():
        artifact = Path(out) / name
        if not artifact.is_file() or sha256(artifact) != digest:
            raise ValueError(f"Completed stage {stage} was modified: {artifact}")
    return True


def finish(out, stage, files):
    write_json(Path(out) / (stage + ".done.json"), {
        "outputs": {str(Path(p).relative_to(out)): sha256(p) for p in files}})


RECIPE_KEYS = ("in_dim", "out_dim", "head_type", "epochs", "episodes_per_epoch",
               "n_way", "k_support", "q_query", "lambda_cons", "lr",
               "weight_decay", "tau", "beta", "learnable_beta", "selection_metric",
               "cache_version", "research_cache_version", "method", "training_complete")


def compare_recipes(records):
    """Missing metadata is unknown, never proof that two recipes match."""
    missing, different = {}, {}
    for key in RECIPE_KEYS:
        absent = [name for name, row in records.items() if row.get(key) is None]
        if absent:
            missing[key] = absent
        values = {name: row.get(key) for name, row in records.items()}
        if len({json.dumps(v, sort_keys=True) for v in values.values() if v is not None}) > 1:
            different[key] = values
    return {"missing": missing, "different": different,
            "metadata_recipe_match": not missing and not different,
            "training_data_identity_verified": False}
