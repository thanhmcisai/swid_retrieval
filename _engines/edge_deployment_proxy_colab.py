# -*- coding: utf-8 -*-
"""
edge_deployment_proxy_colab.py

Lightweight deployment-proxy benchmark for the wood-ID submission.

This script is intentionally separate from final_research.py. It does NOT run
RQ1-RQ3 experiments, does NOT train models, and does NOT write result JSON used
by the main paper. It only measures:

  1) Single-image and small-batch feature-extraction latency after images have
     been decoded/transformed and preloaded as tensors.
  2) Approximate peak GPU memory during inference.
  3) Brute-force cosine-search latency as gallery size grows.

Recommended Colab usage:

    !python edge_deployment_proxy_colab.py

Optional CPU proxy, slower:

    !RUN_CPU_PROXY=1 N_IMAGES=32 python edge_deployment_proxy_colab.py

Outputs:

    /content/drive/MyDrive/NCS/results/paper_reframe/edge_deployment_proxy/
      - edge_proxy_latency.csv
      - edge_proxy_search_scaling.csv
      - edge_proxy_summary.json

Interpretation:
These are deployment proxies, not true edge-device benchmarks. GPU batch-1
latency approximates single-image operation on a server/Colab accelerator.
CPU latency is a rough portability stress test and should not be reported as
Jetson/smartphone performance. Google Drive I/O and image decoding are excluded
from timed model-forward measurements by default.
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm


ROOT_PATH = Path(os.environ.get("ROOT_PATH", "/content/drive/MyDrive/NCS"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT_PATH / "results/paper_reframe"))
CKPT_DIR = Path(os.environ.get("CKPT_DIR", ROOT_PATH / "checkpoints"))
OUT_DIR = RESULTS_DIR / "edge_deployment_proxy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_IMAGES = int(os.environ.get("N_IMAGES", "64"))
WARMUP = int(os.environ.get("WARMUP", "5"))
ITERS = int(os.environ.get("ITERS", "30"))
BATCH_SIZES = [int(x) for x in os.environ.get("BATCH_SIZES", "1,8").split(",") if x.strip()]
RUN_CPU_PROXY = os.environ.get("RUN_CPU_PROXY", "0") == "1"
PRELOAD_TENSORS = os.environ.get("PRELOAD_TENSORS", "1") != "0"
BENCHMARK_MODE = os.environ.get("BENCHMARK_MODE", "gpu_tensor").strip().lower()
SEED = int(os.environ.get("EDGE_PROXY_SEED", "42"))

SCURD_CKPT = (
    RESULTS_DIR
    / "research_directions"
    / os.environ.get("SCURD_CKPT_NAME", "sc_urd_checkpoint_scurd_r01_e20_v2.pt")
)

METHODS = [m.strip() for m in os.environ.get(
    "EDGE_METHODS",
    "ArcFace-557,CE-Full,DINOv2,SC-URD"
).split(",") if m.strip()]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EmbeddingModel(nn.Module):
    def __init__(self, backbone_name="convnext_base", embedding_dim=512, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        return F.normalize(self.head(self.backbone(x)), dim=1)


class CEClassifier(nn.Module):
    def __init__(self, backbone_name="convnext_base", n_classes=954, embedding_dim=512, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        return self.classifier(self.head(self.backbone(x)))

    def get_embedding(self, x, normalize=True):
        feat = self.head(self.backbone(x))
        return F.normalize(feat, dim=1) if normalize else feat


class SCURDResidualHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=512, hidden_dim=1024, beta=0.1, learnable_beta=False):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim, bias=False)
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )
        if learnable_beta:
            beta = float(beta if isinstance(beta, (int, float)) else 0.1)
            beta = min(max(beta, 1e-4), 0.999)
            self.logit_beta = nn.Parameter(torch.logit(torch.tensor(beta, dtype=torch.float32)))
        else:
            self.register_buffer("fixed_beta", torch.tensor(float(beta), dtype=torch.float32))
            self.logit_beta = None

    def beta_value(self):
        if self.logit_beta is None:
            return self.fixed_beta
        return torch.sigmoid(self.logit_beta)

    def forward(self, x):
        base = F.normalize(self.base(x), dim=1)
        delta = F.normalize(self.adapter(x), dim=1)
        beta = self.beta_value().to(x.device)
        return F.normalize(base + beta * delta, dim=1)


class SCURDAnchoredDINOHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=1024, beta=0.05):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, in_dim),
        )
        self.register_buffer("fixed_beta", torch.tensor(float(beta), dtype=torch.float32))

    def beta_value(self):
        return self.fixed_beta

    def forward(self, x):
        base = F.normalize(x, dim=1)
        delta = F.normalize(self.adapter(x), dim=1)
        beta = self.beta_value().to(x.device)
        return F.normalize(base + beta * delta, dim=1)


def image_paths():
    candidates = [
        ROOT_PATH / "ID_images_expanded.csv",
        ROOT_PATH / "OOD_images_expanded.csv",
    ]
    rows = []
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if "file_path" in df.columns:
                rows.extend(df["file_path"].astype(str).tolist())
    rows = [p for p in rows if Path(p).exists()]
    if not rows:
        raise FileNotFoundError(
            "No image paths found. Expected ID_images_expanded.csv or "
            "OOD_images_expanded.csv under ROOT_PATH."
        )
    rng = np.random.RandomState(SEED)
    rng.shuffle(rows)
    return rows[: min(N_IMAGES, len(rows))]


def load_batch(paths, tfm, device):
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(tfm(img))
    return torch.stack(imgs, dim=0).to(device, non_blocking=True)


def preload_batches(paths, tfm, device, batch_size, preload_to_device=True):
    """Decode and transform images before timing.

    If ``preload_to_device`` is True, GPU benchmarks exclude CPU->GPU transfer
    as well as Drive/image-decoding I/O. If False, only Drive/image-decoding is
    excluded and each timing iteration includes host->device transfer.
    """
    batches = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        if len(chunk) != batch_size:
            continue
        imgs = []
        for p in chunk:
            img = Image.open(p).convert("RGB")
            imgs.append(tfm(img))
        x = torch.stack(imgs, dim=0)
        if preload_to_device:
            x = x.to(device, non_blocking=True)
        else:
            x = x.pin_memory() if device.type == "cuda" else x
        batches.append(x)
    return batches


def batch_to_device(batch, device):
    if batch.device == device:
        return batch
    return batch.to(device, non_blocking=True)


def convnext_transform():
    return transforms.Compose([
        transforms.Resize(246, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def dinov2_transform():
    return transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_arcface(device):
    ckpt_path = CKPT_DIR / "metric_convnext_base_arcface.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = EmbeddingModel("convnext_base", 512, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval().to(device), convnext_transform(), "ConvNeXt-B ArcFace embedding"


def load_ce(device):
    ckpt_path = CKPT_DIR / "ce_954sp_convnext_base.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_classes = int(ckpt["metrics"]["n_classes"])
    model = CEClassifier("convnext_base", n_classes=n_classes, embedding_dim=512, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    class Wrapper(nn.Module):
        def __init__(self, ce):
            super().__init__()
            self.ce = ce

        def forward(self, x):
            return self.ce.get_embedding(x, normalize=True)

    return Wrapper(model).eval().to(device), convnext_transform(), "ConvNeXt-B CE feature embedding"


def load_dinov2(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    return model.eval().to(device), dinov2_transform(), "DINOv2 ViT-B/14 embedding"


def load_scurd(device):
    dino, tfm, _ = load_dinov2(device)
    if not SCURD_CKPT.exists():
        raise FileNotFoundError(SCURD_CKPT)
    ckpt = torch.load(SCURD_CKPT, map_location="cpu")
    in_dim = int(ckpt.get("in_dim", 768))
    out_dim = int(ckpt.get("out_dim", 512))
    hidden_dim = int(ckpt.get("hidden_dim", 1024))
    beta = ckpt.get("beta", 0.1)
    state = ckpt["model_state_dict"]
    head_type = ckpt.get("head_type")
    if head_type is None:
        head_type = "anchored_dino" if "base.weight" not in state else "residual"

    if head_type == "anchored_dino":
        beta_init = 0.05 if not isinstance(beta, (int, float)) else float(beta)
        head = SCURDAnchoredDINOHead(in_dim, hidden_dim=hidden_dim, beta=beta_init)
    else:
        learnable_beta = bool(ckpt.get("learnable_beta", False))
        beta_init = 0.1 if learnable_beta or not isinstance(beta, (int, float)) else float(beta)
        head = SCURDResidualHead(
            in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            beta=beta_init,
            learnable_beta=learnable_beta,
        )
    head.load_state_dict(ckpt["model_state_dict"])
    head.eval().to(device)

    class Wrapper(nn.Module):
        def __init__(self, encoder, proj):
            super().__init__()
            self.encoder = encoder
            self.proj = proj

        def forward(self, x):
            feat = self.encoder(x).float()
            return self.proj(feat)

    return Wrapper(dino, head).eval().to(device), tfm, "DINOv2 + SC-URD residual projection"


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_gb(device):
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / 1024**3)


def benchmark_model(name, model, tfm, paths, device, batch_size):
    model.eval()
    preload_to_device = BENCHMARK_MODE != "cpu_tensor_transfer"
    if PRELOAD_TENSORS:
        t0_pre = time.perf_counter()
        batches = preload_batches(paths, tfm, device, batch_size, preload_to_device=preload_to_device)
        synchronize(device)
        preload_sec = time.perf_counter() - t0_pre
    else:
        batches = []
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            if len(chunk) == batch_size:
                batches.append(chunk)
        preload_sec = None
    if not batches:
        return None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for i in range(min(WARMUP, len(batches))):
            if PRELOAD_TENSORS:
                x = batch_to_device(batches[i], device)
            else:
                x = load_batch(batches[i], tfm, device)
            _ = model(x)
        synchronize(device)

        times = []
        n_iter = min(ITERS, len(batches))
        for i in range(n_iter):
            if PRELOAD_TENSORS:
                x = batch_to_device(batches[i % len(batches)], device)
            else:
                x = load_batch(batches[i % len(batches)], tfm, device)
            synchronize(device)
            t0 = time.perf_counter()
            y = model(x)
            synchronize(device)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            _ = y.shape

    per_img = np.asarray(times, dtype=float) / float(batch_size)
    return {
        "method": name,
        "device": str(device),
        "batch_size": int(batch_size),
        "n_images_used": int(n_iter * batch_size),
        "latency_ms_per_image_mean": float(per_img.mean() * 1000.0),
        "latency_ms_per_image_p50": float(np.percentile(per_img, 50) * 1000.0),
        "latency_ms_per_image_p95": float(np.percentile(per_img, 95) * 1000.0),
        "throughput_img_per_sec": float(batch_size / np.mean(times)),
        "peak_gpu_memory_gb": peak_memory_gb(device),
        "preload_tensors": bool(PRELOAD_TENSORS),
        "benchmark_mode": BENCHMARK_MODE,
        "preload_sec": None if preload_sec is None else float(preload_sec),
        "timed_scope": (
            "model_forward_only_tensor_preloaded_on_device"
            if PRELOAD_TENSORS and preload_to_device
            else "model_forward_plus_host_to_device_transfer"
            if PRELOAD_TENSORS
            else "image_decode_transform_transfer_plus_model_forward"
        ),
    }


def benchmark_search(device):
    rows = []
    dims = [512, 768]
    gallery_sizes = [int(x) for x in os.environ.get("GALLERY_SIZES", "1000,5000,10000,50000").split(",")]
    q_count = int(os.environ.get("SEARCH_QUERIES", "1"))
    repeats = int(os.environ.get("SEARCH_REPEATS", "50"))
    rng = torch.Generator(device=device)
    rng.manual_seed(SEED)

    for dim in dims:
        for n_gal in gallery_sizes:
            q = torch.randn(q_count, dim, generator=rng, device=device)
            g = torch.randn(n_gal, dim, generator=rng, device=device)
            q = F.normalize(q, dim=1)
            g = F.normalize(g, dim=1)
            synchronize(device)
            for _ in range(5):
                _ = torch.matmul(q, g.T).max(dim=1)
            synchronize(device)
            times = []
            for _ in range(repeats):
                synchronize(device)
                t0 = time.perf_counter()
                score, idx = torch.matmul(q, g.T).max(dim=1)
                synchronize(device)
                t1 = time.perf_counter()
                times.append(t1 - t0)
                _ = score, idx
            arr = np.asarray(times)
            rows.append({
                "device": str(device),
                "embedding_dim": int(dim),
                "gallery_size": int(n_gal),
                "queries": int(q_count),
                "search_ms_mean": float(arr.mean() * 1000.0),
                "search_ms_p50": float(np.percentile(arr, 50) * 1000.0),
                "search_ms_p95": float(np.percentile(arr, 95) * 1000.0),
            })
            del q, g
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def load_method(name, device):
    if name == "ArcFace-557":
        return load_arcface(device)
    if name == "CE-Full":
        return load_ce(device)
    if name == "DINOv2":
        return load_dinov2(device)
    if name == "SC-URD":
        return load_scurd(device)
    raise ValueError(f"Unknown method: {name}")


def main():
    set_seed(SEED)
    paths = image_paths()
    print(f"ROOT_PATH={ROOT_PATH}")
    print(f"OUT_DIR={OUT_DIR}")
    print(f"Images used={len(paths)}")
    print(f"Methods={METHODS}")
    print(f"Batch sizes={BATCH_SIZES}")
    print(f"PRELOAD_TENSORS={PRELOAD_TENSORS}")
    print(f"BENCHMARK_MODE={BENCHMARK_MODE}")

    devices = []
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    else:
        print("WARNING: CUDA not available; running CPU only.")
    if RUN_CPU_PROXY:
        devices.append(torch.device("cpu"))

    latency_rows = []
    for device in devices:
        print(f"\n=== Device: {device} ===")
        for method in METHODS:
            try:
                print(f"Loading {method}...")
                model, tfm, note = load_method(method, device)
                for bs in BATCH_SIZES:
                    print(f"Benchmark {method}, batch={bs}")
                    row = benchmark_model(method, model, tfm, paths, device, bs)
                    if row is not None:
                        row["note"] = note
                        latency_rows.append(row)
                        print(
                            f"  {row['latency_ms_per_image_mean']:.2f} ms/img "
                            f"p95={row['latency_ms_per_image_p95']:.2f} "
                            f"throughput={row['throughput_img_per_sec']:.2f} img/s"
                        )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                latency_rows.append({
                    "method": method,
                    "device": str(device),
                    "error": repr(exc),
                })
                print(f"ERROR for {method} on {device}: {exc}")

    search_rows = []
    for device in devices:
        print(f"\nSearch scaling on {device}...")
        try:
            search_rows.extend(benchmark_search(device))
        except Exception as exc:
            search_rows.append({"device": str(device), "error": repr(exc)})
            print(f"ERROR in search benchmark on {device}: {exc}")

    latency_df = pd.DataFrame(latency_rows)
    search_df = pd.DataFrame(search_rows)
    latency_path = OUT_DIR / "edge_proxy_latency.csv"
    search_path = OUT_DIR / "edge_proxy_search_scaling.csv"
    latency_df.to_csv(latency_path, index=False)
    search_df.to_csv(search_path, index=False)

    summary = {
        "root_path": str(ROOT_PATH),
        "results_dir": str(RESULTS_DIR),
        "output_dir": str(OUT_DIR),
        "n_images_requested": N_IMAGES,
        "n_images_used": len(paths),
        "warmup": WARMUP,
        "iters": ITERS,
        "batch_sizes": BATCH_SIZES,
        "methods": METHODS,
        "run_cpu_proxy": RUN_CPU_PROXY,
        "preload_tensors": PRELOAD_TENSORS,
        "benchmark_mode": BENCHMARK_MODE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "scurd_ckpt": str(SCURD_CKPT),
        "latency_csv": str(latency_path),
        "search_csv": str(search_path),
        "interpretation": (
            "These are deployment proxies. Batch-1 latency is closer to field use "
            "than batched throughput. Images are preloaded as tensors before timed "
            "model-forward measurements by default, so Google Drive I/O and image "
            "decoding are excluded. True edge/portable deployment requires "
            "device-specific measurement."
        ),
    }
    with open(OUT_DIR / "edge_proxy_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"\nSaved {latency_path}")
    print(f"Saved {search_path}")
    print(f"Saved {OUT_DIR / 'edge_proxy_summary.json'}")
    print("\nLatency preview:")
    print(latency_df)
    print("\nSearch preview:")
    print(search_df.head(20))


if __name__ == "__main__":
    main()
