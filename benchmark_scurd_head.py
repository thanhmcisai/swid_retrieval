"""Benchmark the final SC-URD head loop without writing training checkpoints."""

import importlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from .audit_support import sha256
from .final_scurd_retrain import RECIPE, _meta_path
from .scurd import SCURDResidualHead


def _engine():
    scratch = Path(tempfile.gettempdir()) / "swid_scurd_benchmark"
    previous = {key: os.environ.get(key) for key in ("RESULTS_DIR", "OUT_DIR")}
    os.environ["RESULTS_DIR"] = str(scratch)
    os.environ["OUT_DIR"] = str(scratch / "engine")
    try:
        return importlib.import_module(
            "swid_retrieval._engines.variance_retrieval_evidence_colab")
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def benchmark_case(engine, weak, strong, labels, device, precomputed, steps, warmup):
    import torch
    import torch.nn.functional as F

    n_way = int(RECIPE["SCURD_N_WAY"])
    k_support = int(RECIPE["SCURD_K_SUPPORT"])
    q_query = int(RECIPE["SCURD_Q_QUERY"])
    tau = float(RECIPE["SCURD_TAU"])
    lambda_cons = float(RECIPE["SCURD_TRAIN_LAMBDA_CONS"])
    pool = engine.scurd_episode_pool(labels, k_support, q_query, n_way)
    rng = np.random.RandomState(42)
    total = steps + warmup

    def next_episode():
        return engine.scurd_episode_indices(
            labels, n_way, k_support, q_query, rng, pool=pool)

    precompute_s = 0.0
    episodes = None
    if precomputed:
        start = time.perf_counter()
        episodes = [next_episode() for _ in range(total)]
        precompute_s = time.perf_counter() - start

    start = time.perf_counter()
    train_w = torch.as_tensor(weak, dtype=torch.float32, device=device)
    train_s = torch.as_tensor(strong, dtype=torch.float32, device=device)
    preload_s = time.perf_counter() - start
    torch.manual_seed(42)
    model = SCURDResidualHead(weak.shape[1], beta=float(RECIPE["SCURD_BETA"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(RECIPE["SCURD_TRAIN_LR"]),
        weight_decay=float(RECIPE["SCURD_WEIGHT_DECAY"]))

    sample_s = 0.0
    metrics = []
    wall_start = None
    for i in range(total):
        if i == warmup:
            if device == "cuda":
                torch.cuda.synchronize()
            wall_start = time.perf_counter()
            sample_s = 0.0
            metrics.clear()
        if episodes is None:
            sample_start = time.perf_counter()
            sup_idx, qry_idx, ys_np, yq_np = next_episode()
            if i >= warmup:
                sample_s += time.perf_counter() - sample_start
        else:
            sup_idx, qry_idx, ys_np, yq_np = episodes[i]

        sup_idx = torch.as_tensor(sup_idx, dtype=torch.long, device=device)
        qry_idx = torch.as_tensor(qry_idx, dtype=torch.long, device=device)
        ys = torch.as_tensor(ys_np, dtype=torch.long, device=device)
        yq = torch.as_tensor(yq_np, dtype=torch.long, device=device)
        support_z = model(train_w[sup_idx])
        weak_z = model(train_w[qry_idx])
        strong_z = model(train_s[qry_idx])
        n_way_eff = int(yq_np.max()) + 1
        logits_w = engine.urd_logits(
            weak_z, support_z, ys, n_way_eff, tau, support_complete=True)
        logits_s = engine.urd_logits(
            strong_z, support_z, ys, n_way_eff, tau, support_complete=True)
        loss_cls = F.cross_entropy(logits_w, yq)
        pw = F.log_softmax(logits_w, dim=1)
        ps = F.log_softmax(logits_s, dim=1)
        loss_cons = 0.5 * (
            F.kl_div(pw, ps.exp(), reduction="batchmean")
            + F.kl_div(ps, pw.exp(), reduction="batchmean"))
        loss = loss_cls + lambda_cons * loss_cons
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i >= warmup:
            with torch.no_grad():
                acc = (logits_w.argmax(1) == yq).float().mean()
                metrics.append(torch.stack((loss, loss_cls, loss_cons, acc)))

    epoch_metrics = torch.stack(metrics).cpu().numpy().astype(np.float64)
    if device == "cuda":
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - wall_start
    if not np.isfinite(epoch_metrics).all():
        raise ValueError("Non-finite SC-URD benchmark metrics")
    return {
        "device": device,
        "sampling": "precomputed" if precomputed else "online",
        "steps": steps,
        "warmup": warmup,
        "wall_s": wall_s,
        "ms_per_episode": 1000 * wall_s / steps,
        "sampling_ms_per_episode": 1000 * sample_s / steps,
        "precompute_s": precompute_s,
        "preload_s": preload_s,
        "ms_per_episode_including_precompute": 1000 * (wall_s + precompute_s) / steps,
        "mean_loss": float(np.mean(epoch_metrics[:, 0])),
    }


def run():
    import torch

    root = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS"))
    run_root = Path(os.environ.get(
        "FINAL_AUDIT_RUN_ROOT",
        root / "results" / "paper_reframe_full954_retrained_ce_corrected_public"))
    meta_path = _meta_path(root, run_root)
    steps = int(os.environ.get("BENCH_SCURD_STEPS", "100"))
    warmup = int(os.environ.get("BENCH_SCURD_WARMUP", "10"))
    if steps < 1 or warmup < 0:
        raise ValueError("BENCH_SCURD_STEPS must be positive and warmup nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this benchmark on the Colab GPU runtime to compare CPU and CUDA")

    engine = _engine()
    with np.load(meta_path, allow_pickle=False) as data:
        weak = data["train_weak"].astype(np.float32)
        strong = data["train_strong"].astype(np.float32)
        labels = np.asarray([engine.canonical_label(x) for x in data["train_labels"]])
    if weak.shape != strong.shape or len(labels) != len(weak):
        raise ValueError("Meta-cache features and labels have inconsistent shapes")

    cpu_max_path = Path("/sys/fs/cgroup/cpu.max")
    cpu_quota = None
    if cpu_max_path.is_file():
        quota, period = cpu_max_path.read_text().split()[:2]
        if quota != "max":
            cpu_quota = float(quota) / float(period)

    report = {
        "meta_cache": str(meta_path),
        "feature_shape": list(weak.shape),
        "training_engine_sha256": sha256(Path(engine.__file__)),
        "recipe": RECIPE,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "cpu_quota_cores": cpu_quota,
        "torch_cpu_threads": torch.get_num_threads(),
        "note": "Timing only; no checkpoints. CPU is measured on the GPU VM, not a standard Colab VM.",
        "results": [],
    }
    for device in ("cpu", "cuda"):
        for precomputed in (False, True):
            row = benchmark_case(
                engine, weak, strong, labels, device, precomputed, steps, warmup)
            report["results"].append(row)
            print(f"{device:4s} {row['sampling']:11s}: "
                  f"{row['ms_per_episode']:.2f} ms/episode "
                  f"(sampling {row['sampling_ms_per_episode']:.2f} ms; "
                  f"including precompute {row['ms_per_episode_including_precompute']:.2f} ms)",
                  flush=True)

    out = Path(os.environ.get("BENCH_SCURD_OUT", "/content/scurd_head_benchmark.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Benchmark saved: {out}", flush=True)
    return report


if __name__ == "__main__":
    run()
