# -*- coding: utf-8 -*-
"""Multi-scale patch extraction + quality filtering.

Lifted verbatim from old_version_claim_and_process_data/crop.py. For each full
microscopy image: generate hybrid linear+geometric patch scales, tile each scale,
resize to BASE_SIZE, and keep only patches passing the quality gates
(black-ratio / crack / Laplacian sharpness / entropy). Writes
  {out}/{image_id}/scale_{ps}/patch_{ps}_{i}_{j}.jpg
and a merged metadata.csv. CPU-parallel (ProcessPoolExecutor); incremental
(skips image_ids already present). No GPU.
"""

import concurrent.futures
import math
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from . import config as D

Image.MAX_IMAGE_PIXELS = None


# ── Quality filters ──────────────────────────────────────────────────────────
def is_mostly_black(img_np, threshold_ratio=0.5, black_pixel_thresh=30):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    black = np.sum(gray < black_pixel_thresh)
    ratio = black / (img_np.shape[0] * img_np.shape[1])
    return ratio > threshold_ratio, ratio


def has_crack(img_np, min_contour_area=500, aspect_ratio_thresh=5.0):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(cnt)
            ar = float(w) / h if h != 0 else 0
            inv = float(h) / w if w != 0 else 0
            if ar > aspect_ratio_thresh or inv > aspect_ratio_thresh:
                return True, (area, max(ar, inv))
    return False, None


def compute_quality_metrics(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist /= hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return sharpness, entropy


# ── Scale generation + tiling ────────────────────────────────────────────────
def auto_patch_sizes(image_size, base_size=256, linear_thresh=1024, factor=1.5):
    """Hybrid linear (step base_size up to linear_thresh) + geometric (×factor)
    patch sizes; robust fallback for narrow images."""
    W, H = image_size
    min_side = min(W, H)
    linear_sizes = list(range(base_size, min(min_side, linear_thresh) + 1, base_size))
    geometric_sizes = []
    if min_side > linear_thresh and linear_sizes:
        cur = float(linear_sizes[-1]) * factor
        while cur <= min_side:
            geometric_sizes.append(int(round(cur)))
            cur *= factor
    patch_sizes = sorted(set(linear_sizes + geometric_sizes))
    if len(patch_sizes) <= 1:
        fallback = [min_side] if min_side < base_size else [base_size, min_side]
        patch_sizes = sorted(set(fallback))
    return patch_sizes, (W, H)


def crop_and_filter_image(image_path, save_root, base_size=256,
                          sharpness_thresh=5.0, entropy_thresh=3.5,
                          black_ratio_thresh=0.5, crack_aspect_ratio_thresh=5.0):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:  # noqa: BLE001
        return None, f"Error opening {os.path.basename(image_path)}: {e}"

    basename = os.path.splitext(os.path.basename(image_path))[0]
    patch_sizes, (W, H) = auto_patch_sizes(img.size, base_size)
    if not patch_sizes:
        return [], None

    records = []
    image_save_root = os.path.join(save_root, basename)
    for ps in patch_sizes:
        scale_dir = os.path.join(image_save_root, f"scale_{ps}")
        os.makedirs(scale_dir, exist_ok=True)
        n_x, n_y = math.ceil(W / ps), math.ceil(H / ps)
        step_x = (W - ps) / (n_x - 1) if n_x > 1 else 0
        step_y = (H - ps) / (n_y - 1) if n_y > 1 else 0
        for idx in range(n_x * n_y):
            i, j = idx // n_y, idx % n_y
            left, top = int(i * step_x), int(j * step_y)
            patch = img.crop((left, top, min(left + ps, W), min(top + ps, H)))
            if ps != base_size:
                patch = patch.resize((base_size, base_size), Image.Resampling.LANCZOS)
            patch_np = np.array(patch)
            is_black, black_ratio = is_mostly_black(patch_np, threshold_ratio=black_ratio_thresh)
            is_cracked, crack_info = has_crack(patch_np, aspect_ratio_thresh=crack_aspect_ratio_thresh)
            sharpness, entropy = compute_quality_metrics(patch_np)
            status, reason = "kept", "good"
            if is_black:
                status, reason = "removed", f"black_ratio_{black_ratio:.2f}"
            elif is_cracked:
                status, reason = "removed", f"crack_detected_area_{int(crack_info[0])}"
            elif sharpness < sharpness_thresh:
                status, reason = "removed", f"low_sharpness_{sharpness:.2f}"
            elif entropy < entropy_thresh:
                status, reason = "removed", f"low_entropy_{entropy:.2f}"
            relative_path = ""
            if status == "kept":
                fn = f"patch_{ps}_{i}_{j}.jpg"
                patch.save(os.path.join(scale_dir, fn), quality=95)
                relative_path = os.path.join(basename, f"scale_{ps}", fn)
            records.append({"image_id": basename, "patch_size": ps, "i": i, "j": j,
                            "relative_path": relative_path, "sharpness": round(sharpness, 2),
                            "entropy": round(entropy, 2), "status": status, "reason": reason})
    return records, None


def process_folder_parallel(input_folder, output_folder, base_size=256):
    """Crop every image under input_folder in parallel, skipping already-processed
    image_ids, merging into output_folder/metadata.csv."""
    os.makedirs(output_folder, exist_ok=True)
    all_images = [f for f in sorted(os.listdir(input_folder))
                  if os.path.splitext(f)[1].lower() in D.IMAGE_EXTENSIONS]
    print(f"Found {len(all_images)} images in {input_folder}")
    to_process, skipped = [], 0
    for img_name in all_images:
        basename = os.path.splitext(img_name)[0]
        if os.path.isdir(os.path.join(output_folder, basename)):
            skipped += 1
            continue
        to_process.append(os.path.join(input_folder, img_name))
    if skipped:
        print(f"⏩ Skipped {skipped} already-processed images.")
    if not to_process:
        print("✅ No new images to process.")
        return

    all_metadata, max_workers = [], D.num_workers()
    print(f"🚀 Cropping {len(to_process)} images on {max_workers} cores...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(crop_and_filter_image, p, output_folder, base_size,
                          D.SHARPNESS_THRESH, D.ENTROPY_THRESH, D.BLACK_RATIO_THRESH,
                          D.CRACK_ASPECT_RATIO_THRESH): p for p in to_process}
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(to_process), desc="Crop"):
            records, err = fut.result()
            if err:
                print(err)
            if records:
                all_metadata.extend(records)

    csv_path = os.path.join(output_folder, "metadata.csv")
    new_df = pd.DataFrame(all_metadata)
    if os.path.exists(csv_path):
        print(f"🔄 Merging with existing {csv_path}")
        new_df = pd.concat([pd.read_csv(csv_path), new_df], ignore_index=True)
    new_df.to_csv(csv_path, index=False)
    kept = (new_df["status"] == "kept").sum()
    print(f"✅ Crop done: {len(new_df)} patches, {kept} kept → {csv_path}")


def run():
    process_folder_parallel(str(D.RAW_IMAGE_DIR), str(D.PATCH_OUT_DIR), D.BASE_SIZE)


if __name__ == "__main__":
    run()
