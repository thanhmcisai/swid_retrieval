# Full-image ArcFace, contrastive, and QKV ablations

This study visits every one of the 124,577 meta-train images as a query once
per epoch. It uses no public-ID, meta-test, or 954-label CE training images.
Every arm starts from the same DINOv2-S checkpoint (source seed 43, including
the runs labelled training seeds 42 and 44) and uses the same paired
reference plan. A reference is another image of the same species; a different
source scan and crop scale are preferred when available. Same-scan fallbacks
are counted in each epoch's pair audit.

| Arm | Additional objective or sampling |
| --- | --- |
| `arcface` | 557-way ArcFace control |
| `arcface_qkv` | ArcFace plus in-batch QKV class-ranking loss |
| `arcface_pair_nce` | ArcFace plus query-to-paired-reference InfoNCE; other same-class references are excluded from the negative denominator |
| `arcface_supcon` | ArcFace plus symmetric supervised contrastive loss, using the control batch sampler |
| `arcface_packed` | ArcFace only, using the multi-positive batch sampler |
| `arcface_multipos` | ArcFace plus supervised contrastive loss over all query/reference views; pairs of query images from one species are packed into each batch where possible, preferring different scans and scales |
| `arcface_multipos_qkv` | Multi-positive arm plus the in-batch QKV loss |
| `arcface_pair_nce_random` | Matched pair-InfoNCE control: same scan-disjoint preference, but no cross-scale preference when choosing a positive reference |

All objectives backpropagate through the image encoder. The QKV training loss
competes with the distinct reference classes in a batch, not all 557 classes;
the ArcFace classifier provides 557-way discrimination in every arm. The
multi-positive sampler has fewer distinct in-batch negative species than the
control sampler. `arcface_packed` isolates this sampler effect; `arcface_supcon`
isolates the contrastive objective without packing. Candidate-class and
positive-view counts are logged. No stale feature bank is used.

Selection uses the mean R@1 across two scan-disjoint meta-val folds at 57x5,
57x1, and 637x1. After training, `validate` exports per-query predictions for
these three standard galleries. It also compares K=1/3/5/10 with nested
scan-hash and scan/scale-diverse reference selection. The same 285 queries per
fold are used across K and policies. Both global and QKV scoring are reported
for QKV arms, as are candidate recall@32/@128 and R@1 conditional on candidate
inclusion. `validate` also scores the unchanged source encoder with global
prototype retrieval on the same queries and reference selections. The
954-species and held-out meta-test datasets are not used here.

This is full-dataset supervised metric training in the style of face
recognition: every meta-train image is a query once per epoch, and every
update uses a 557-way ArcFace classifier. It is **not** an N-way/K-shot
episodic training loop. Paired-reference InfoNCE, supervised contrastive, and
QKV are optional auxiliary objectives over the current minibatch. The
classifier is not used at deployment. All arms fine-tune the same previously
trained DINOv2-S source backbone; seeds here do not replicate its pretraining.

Run from Colab after mounting Drive. Use a **new** output folder: changes to
the runner are part of checkpoint provenance and old study folders cannot be
resumed with this version.

```python
%cd /content/drive/MyDrive/NCS
!git -C swid_retrieval pull

import os, sys, runpy
sys.path.insert(0, "/content/drive/MyDrive/NCS")
for name in list(sys.modules):
    if name.startswith("swid_retrieval"):
        del sys.modules[name]
for key in list(os.environ):
    if key.startswith("RUN_") or key.startswith("SKIP_"):
        os.environ.pop(key, None)
os.environ.update({
    "ROOT_PATH": "/content/drive/MyDrive/NCS",
    "DEVICE": "cuda",
    "RUN_WOOD_FULL_IMAGE_TRAIN": "1",
    "WOOD_FULL_MODE": "preflight",
    "WOOD_FULL_OUT": "/content/drive/MyDrive/NCS/results/wood_full_image_ablation_v3",
    "WOOD_FULL_SOURCE_OUT": "/content/drive/MyDrive/NCS/results/dinov2s_retrieval_matched_v1",
    "WOOD_FULL_SOURCE_SEED": "43",
    "WOOD_FULL_ARMS": "arcface,arcface_qkv,arcface_pair_nce,arcface_pair_nce_random,arcface_supcon,arcface_packed,arcface_multipos,arcface_multipos_qkv",
    "WOOD_FULL_SEEDS": "42,43,44",
    "WOOD_FULL_EPOCHS": "3",
    "WOOD_FULL_BENCH_STEPS": "3",
    "WOOD_FULL_PRELOAD": "1",
    "WOOD_FULL_PRELOAD_WORKERS": "8",
    "WOOD_FULL_CONTRASTIVE_WEIGHT": "0.2",
    "WOOD_FULL_CONTRASTIVE_TEMPERATURE": "0.07",
    "WOOD_FULL_QKV_WEIGHT": "0.5",
})
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
```

Benchmark on real meta-train image pairs before paying for full training. This
benchmarks ArcFace+QKV, the heavier training branch, with one warm-up and three
timed batches per configuration. It records data wait, compute time, throughput,
and reserved VRAM; it does not write training checkpoints. A 12-GiB safety
ceiling avoids selecting a configuration too close to T4 capacity. The direct
path keeps all microbatch graphs until backward; the replay path uses two
encoder forwards to reduce peak memory. FP16 gradient scaling is enabled for
both paths. The benchmark is short, so inspect its timings before accepting
the recommendation, especially if Drive I/O is variable.

```python
import json
from pathlib import Path

os.environ["WOOD_FULL_MODE"] = "benchmark"
_ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
report = json.loads((Path(os.environ["WOOD_FULL_OUT"]) / "hardware_benchmark.json").read_text())
choice = report["recommendation"]
print("Chosen T4 setting:", choice)
os.environ.update({
    "WOOD_FULL_BATCH": str(choice["batch"]),
    "WOOD_FULL_MICROBATCH": str(choice["microbatch"]),
    "WOOD_FULL_STEP_MODE": choice["step_mode"],
    "WOOD_FULL_WORKERS": str(choice["workers"]),
})
```

Then run preflight and the entire study in one cell. Re-running resumes completed training
epochs. It leaves checkpoints on Drive and exports only small reports:

```python
for mode in ("preflight", "train", "validate", "export"):
    os.environ["WOOD_FULL_MODE"] = mode
    _ = runpy.run_module("swid_retrieval.run_overnight", run_name="__main__")
print(os.environ["WOOD_FULL_OUT"] + "/wood_full_image_ablation_review.zip")
```

Eight arms times three seeds is expensive on a T4. For a throughput pilot, use
only `arcface_pair_nce,arcface_multipos` and seed `43` in a separate output
folder. Do not change the source, losses, batch size, or other recipe values
inside an existing folder; the provenance guard rejects such changes. The
review ZIP contains JSON/CSV reports, not weights or embedding arrays.
These are fine-tuning seeds from a shared source encoder, not independent
backbone-pretraining replications. Do not select a loss, scorer, or reference
policy using meta-test or public-ID.
