# -*- coding: utf-8 -*-
"""Deployment cost proxy — gallery search-latency scaling (tab:inference_cost).

Lightweight, cache-only: measures brute-force cosine nearest-neighbour search
latency as the gallery grows, per embedding dim, from cached embeddings (no model
load). Single-image FORWARD latency + peak memory are produced by the edge proxy
(tab:edge_proxy); gallery-enrollment and joint-retrain training cost remain in the
edge/CE-finetune paths. Output → deployment_search_cost.json.
"""

import json
import os
import time

import numpy as np

from . import registry as R


def run(M, ctx, out_dir):
    rng = np.random.RandomState(42)
    sizes = [1000, 5000, 20000, 100000]
    rows = []
    for name in ["ArcFace-557", "DINOv2", "SC-URD"]:
        if name not in M:
            continue
        gal = R._norm(M[name]["gal"][0])
        dim = gal.shape[1]
        q = R._norm(M[name]["id"][:64])
        for n in sizes:
            if n <= len(gal):
                idx = rng.choice(len(gal), n, replace=False); G = gal[idx]
            else:  # tile up to the target size
                reps = int(np.ceil(n / len(gal)))
                G = np.tile(gal, (reps, 1))[:n]
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                _ = (q @ G.T).argmax(axis=1)
                ts.append((time.perf_counter() - t0) * 1000.0 / len(q))
            rows.append({"method": name, "embedding_dim": int(dim), "gallery_size": int(n),
                         "search_ms_per_query_mean": float(np.mean(ts)),
                         "search_ms_per_query_p95": float(np.percentile(ts, 95))})
        print(f"  search cost {name}: dim={dim}")
    out = {"note": "Brute-force cosine search latency vs gallery size (CPU/GPU host). "
                   "Forward latency + peak memory: see edge_deployment_proxy (tab:edge_proxy).",
           "search_scaling": rows}
    os.makedirs(str(out_dir), exist_ok=True)
    path = os.path.join(str(out_dir), "deployment_search_cost.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"saved {path}")
    return out
