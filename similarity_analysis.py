#!/usr/bin/env python3
"""Measure feature-space neighbors and image similarity for selected synthetic samples.

Private input manifests and per-image results contain image paths and identifiers.
Keep them with the clinical data on the approved remote; publish only aggregate
tables and identifier-free figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
PIXEL_COUNTS = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def letterbox(path: str | Path, size: int) -> np.ndarray:
    """EXIF-correct, aspect-preserving RGB resize with reflect padding."""
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    width, height = image.size
    scale = min(size / max(width, 1), size / max(height, 1))
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(target, LANCZOS)
    left = (size - target[0]) // 2
    top = (size - target[1]) // 2
    right = size - target[0] - left
    bottom = size - target[1] - top
    array = np.asarray(resized, dtype=np.uint8)
    padded = np.pad(array, ((top, bottom), (left, right), (0, 0)), mode="reflect")
    return np.asarray(padded, dtype=np.uint8)


def difference_hash(array: np.ndarray) -> int:
    gray = Image.fromarray(array, mode="RGB").convert("L").resize((9, 8), LANCZOS)
    values = np.asarray(gray, dtype=np.uint8)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.reshape(-1):
        result = (result << 1) | int(bit)
    return result


def _box_mean(array: np.ndarray, window: int = 11) -> np.ndarray:
    """Mean filter from an integral image, without an extra image dependency."""
    radius = window // 2
    padded = np.pad(array, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    sums = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return sums / float(window * window)


def structural_similarity(gray_a: np.ndarray, gray_b: np.ndarray, window: int = 11) -> float:
    """Mean SSIM with an 11x11 uniform window and 8-bit luminance range."""
    a = gray_a.astype(np.float64, copy=False)
    b = gray_b.astype(np.float64, copy=False)
    mean_a, mean_b = _box_mean(a, window), _box_mean(b, window)
    variance_a = np.maximum(_box_mean(a * a, window) - mean_a * mean_a, 0.0)
    variance_b = np.maximum(_box_mean(b * b, window) - mean_b * mean_b, 0.0)
    covariance = _box_mean(a * b, window) - mean_a * mean_b
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * covariance + c2)
    denominator = (mean_a * mean_a + mean_b * mean_b + c1) * (variance_a + variance_b + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def load_image_features(paths: list[str], model, torch, size: int, batch_size: int,
                        device, workers: int, description: str) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Read each image once, retaining gray pixels and hashes for later checks."""
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    features: list[np.ndarray] = []
    gray_images = np.empty((len(paths), size, size), dtype=np.uint8)
    pixel_hashes: list[str] = []
    dhashes = np.empty(len(paths), dtype=np.uint64)
    model.eval()
    print(f"Encoding {len(paths):,} {description} images on {device}...", flush=True)
    started = time.monotonic()
    batch_count = (len(paths) + batch_size - 1) // batch_size
    with ThreadPoolExecutor(max_workers=workers) as executor, torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(paths), batch_size), start=1):
            batch_paths = paths[start:start + batch_size]
            images = list(executor.map(lambda path: letterbox(path, size), batch_paths))
            for offset, image in enumerate(images):
                index = start + offset
                gray_images[index] = np.asarray(Image.fromarray(image).convert("L"), dtype=np.uint8)
                pixel_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
                dhashes[index] = difference_hash(image)
            array = np.stack(images).transpose(0, 3, 1, 2).copy()
            values = torch.from_numpy(array).to(dtype=torch.float32).div_(255.0)
            values.sub_(mean).div_(std)
            batch = model(values.to(device, non_blocking=True))
            batch = torch.nn.functional.normalize(batch, p=2, dim=1)
            features.append(batch.cpu().numpy().astype(np.float32, copy=False))
            if batch_index % 50 == 0 or batch_index == batch_count:
                print(f"{description}: {min(start + len(batch_paths), len(paths)):,}/{len(paths):,} images; {time.monotonic() - started:.1f}s", flush=True)
    return np.concatenate(features, axis=0), gray_images, pixel_hashes, dhashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path,
                        help="Private selected_synthetic_samples.csv")
    parser.add_argument("--training-reference", required=True, type=Path,
                        help="Private reference CSV containing generator-fit and monitor images")
    parser.add_argument("--weights", required=True, type=Path,
                        help="Cached torchvision ResNet50 ImageNet V1 state dictionary")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12,
                        help="Concurrent image readers for remote storage")
    parser.add_argument("--device", default="cpu", help="cpu or a torch device such as cuda:0")
    args = parser.parse_args()

    if args.image_size < 32 or min(args.batch_size, args.threads, args.workers) < 1:
        parser.error("image size must be >=32; batch size, threads, and workers must be positive")
    if not args.weights.is_file():
        parser.error(f"ResNet50 weights are not present at {args.weights}; provide a cached file")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    import torch
    from torchvision import models

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(min(args.threads, 2))
    except RuntimeError:
        pass
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(f"CUDA device {args.device} was requested but CUDA is unavailable")
    model = models.resnet50(weights=None)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.fc = torch.nn.Identity()
    model.to(device).eval()

    samples = pd.read_csv(args.samples)
    required_samples = {
        "generation_method", "seed", "selected_route", "synthetic_id",
        "synthetic_label", "synthetic_path",
    }
    missing = sorted(required_samples - set(samples.columns))
    if missing:
        parser.error(f"sample manifest is missing columns: {missing}")
    samples["generation_method"] = samples["generation_method"].astype(str).str.lower()
    samples["seed"] = samples["seed"].astype(str)
    samples["synthetic_label"] = pd.to_numeric(samples["synthetic_label"], errors="raise").astype(int)
    samples["synthetic_path"] = samples["synthetic_path"].astype(str)
    if samples.empty or not set(samples["generation_method"].unique()).issubset({"gan", "diffusion"}):
        parser.error("sample manifest must contain GAN and/or diffusion rows")
    if samples.duplicated(["generation_method", "seed", "synthetic_id"]).any() or samples["synthetic_path"].duplicated().any():
        parser.error("sample IDs and image paths must be unique")
    if not set(samples["synthetic_label"].unique()).issubset({0, 1}):
        parser.error("labels must be encoded as 0 or 1")

    partition = pd.read_csv(args.training_reference, dtype={"patient_group": str, "image_id": str, "absolute_path": str})
    required_reference = {"generator_partition", "patient_group", "image_id", "label", "absolute_path"}
    missing = sorted(required_reference - set(partition.columns))
    if missing:
        parser.error(f"training reference is missing columns: {missing}")
    reference = partition.loc[
        partition["generator_partition"].astype(str).isin({"fit", "monitor"})
    ].copy()
    reference["label"] = pd.to_numeric(reference["label"], errors="raise").astype(int)
    if reference.empty or reference["image_id"].duplicated().any():
        parser.error("training reference is empty or contains duplicate image IDs")
    if len(reference) != 1695:
        parser.error(f"expected 1,695 real training references, found {len(reference)}")
    if reference["absolute_path"].duplicated().any():
        parser.error("training reference contains duplicate image paths")
    if set(reference["generator_partition"].astype(str)) != {"fit", "monitor"}:
        parser.error("both generator-fit and checkpoint-monitor images must be present")
    partition_counts = reference["generator_partition"].astype(str).value_counts().to_dict()
    if partition_counts != {"fit": 1439, "monitor": 256}:
        parser.error(f"unexpected real training partition sizes: {partition_counts}")
    if not set(reference["label"].unique()).issubset({0, 1}):
        parser.error("reference labels must be encoded as 0 or 1")
    fit_groups = set(reference.loc[reference["generator_partition"].eq("fit"), "patient_group"])
    monitor_groups = set(reference.loc[reference["generator_partition"].eq("monitor"), "patient_group"])
    if fit_groups & monitor_groups:
        parser.error("patient groups cross generator-fit and checkpoint-monitor partitions")
    samples = samples.sort_values(
        ["generation_method", "seed", "synthetic_label", "synthetic_id"], kind="stable"
    ).reset_index(drop=True)
    expected_per_route = {"gan": 1695, "diffusion": 5085}
    expected_seeds = {"17", "42", "2024", "2025", "3407"}
    if set(samples["seed"]) != expected_seeds:
        parser.error(f"expected the five final seeds {sorted(expected_seeds)}, found {sorted(set(samples['seed']))}")
    route_counts = samples.groupby(["generation_method", "seed"], sort=True).size()
    for method, expected_count in expected_per_route.items():
        for seed in sorted(expected_seeds):
            if int(route_counts.get((method, seed), 0)) != expected_count:
                parser.error(f"{method} seed {seed}: expected {expected_count} synthetic images")
    expected_labels = {"gan": {0: 1145, 1: 550}, "diffusion": {0: 3435, 1: 1650}}
    for (method, seed), group in samples.groupby(["generation_method", "seed"], sort=True):
        observed = {int(k): int(v) for k, v in group["synthetic_label"].value_counts().sort_index().items()}
        if observed != expected_labels[str(method)]:
            parser.error(f"{method} seed {seed}: unexpected synthetic class counts {observed}")

    reference = reference.sort_values(["generator_partition", "image_id"], kind="stable").reset_index(drop=True)
    reference_paths = reference["absolute_path"].astype(str).tolist()
    sample_paths = samples["synthetic_path"].tolist()
    real_features, real_gray, real_hashes, real_dhashes = load_image_features(
        reference_paths, model, torch, args.image_size, args.batch_size, device, args.workers, "real training-reference",
    )
    synthetic_features, synthetic_gray, synthetic_hashes, synthetic_dhashes = load_image_features(
        sample_paths, model, torch, args.image_size, args.batch_size, device, args.workers, "synthetic",
    )
    real_features_tensor = torch.as_tensor(real_features, dtype=torch.float32, device=device)
    synthetic_features_tensor = torch.as_tensor(synthetic_features, dtype=torch.float32, device=device)
    real_features_tensor = torch.nn.functional.normalize(real_features_tensor, p=2, dim=1)
    synthetic_features_tensor = torch.nn.functional.normalize(synthetic_features_tensor, p=2, dim=1)
    real_labels = reference["label"].astype(int).to_numpy()
    synthetic_labels = samples["synthetic_label"].to_numpy()
    real_ids = reference["image_id"].astype(str).to_numpy()
    real_groups = reference["patient_group"].astype(str).to_numpy()
    real_parts = reference["generator_partition"].astype(str).to_numpy()

    best_indices = np.empty(len(samples), dtype=np.int32)
    best_scores = np.empty(len(samples), dtype=np.float32)
    same_class_indices = np.empty(len(samples), dtype=np.int32)
    same_class_scores = np.empty(len(samples), dtype=np.float32)
    print("Searching feature-space nearest neighbors...", flush=True)
    with torch.inference_mode():
        for start in range(0, len(samples), 512):
            stop = min(start + 512, len(samples))
            scores = synthetic_features_tensor[start:stop] @ real_features_tensor.T
            values, indices = scores.max(dim=1)
            best_scores[start:stop] = values.cpu().numpy()
            best_indices[start:stop] = indices.cpu().numpy()
            labels = torch.as_tensor(synthetic_labels[start:stop], dtype=torch.long, device=device)
            same_class = torch.as_tensor(real_labels, dtype=torch.long, device=device)[None, :] == labels[:, None]
            class_values, class_indices = scores.masked_fill(~same_class, -torch.inf).max(dim=1)
            same_class_scores[start:stop] = class_values.cpu().numpy()
            same_class_indices[start:stop] = class_indices.cpu().numpy()
            if stop == len(samples) or stop % 5120 == 0:
                print(f"Nearest-neighbor search: {stop:,}/{len(samples):,} samples", flush=True)

    ssim_values: list[float] = []
    window = torch.ones((5, 1, 11, 11), dtype=torch.float32, device=device) / 121.0
    print("Computing SSIM for each feature-nearest pair...", flush=True)
    with torch.inference_mode():
        for start in range(0, len(samples), 64):
            stop = min(start + 64, len(samples))
            synth = torch.from_numpy(synthetic_gray[start:stop].copy()).to(device=device, dtype=torch.float32).div_(255.0).unsqueeze(1)
            real = torch.from_numpy(real_gray[best_indices[start:stop]].copy()).to(device=device, dtype=torch.float32).div_(255.0).unsqueeze(1)
            moments_input = torch.cat((synth, real, synth * synth, real * real, synth * real), dim=1)
            moments = torch.nn.functional.conv2d(moments_input, window, groups=5)
            mean_x, mean_y, mean_x2, mean_y2, mean_xy = moments.unbind(dim=1)
            variance_x = torch.clamp(mean_x2 - mean_x.square(), min=0.0)
            variance_y = torch.clamp(mean_y2 - mean_y.square(), min=0.0)
            covariance = mean_xy - mean_x * mean_y
            c1, c2 = 0.01**2, 0.03**2
            ssim_map = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
                (mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2).clamp_min(1e-12)
            )
            ssim_values.extend(ssim_map.mean(dim=(1, 2)).cpu().tolist())
            if stop == len(samples) or stop % 3200 == 0:
                print(f"SSIM: {stop:,}/{len(samples):,} pairs", flush=True)

    min_dhash = np.empty(len(samples), dtype=np.uint8)
    min_dhash_indices = np.empty(len(samples), dtype=np.int32)
    print("Screening exact pixels and 64-bit difference hashes...", flush=True)
    for start in range(0, len(samples), 512):
        stop = min(start + 512, len(samples))
        xor = np.bitwise_xor(synthetic_dhashes[start:stop, None], real_dhashes[None, :])
        byte_view = xor.view(np.uint8).reshape(stop - start, len(reference), 8)
        distances = PIXEL_COUNTS[byte_view].sum(axis=2, dtype=np.uint16)
        min_dhash_indices[start:stop] = distances.argmin(axis=1)
        min_dhash[start:stop] = distances.min(axis=1).astype(np.uint8)
        if stop == len(samples) or stop % 5120 == 0:
            print(f"Difference-hash screening: {stop:,}/{len(samples):,} samples", flush=True)

    real_hash_set = set(real_hashes)
    rows: list[dict] = []
    for index, sample in samples.iterrows():
        best = int(best_indices[index])
        same_class = int(same_class_indices[index])
        hash_neighbor = int(min_dhash_indices[index])
        rows.append({
            "generation_method": str(sample["generation_method"]),
            "seed": str(sample["seed"]),
            "selected_route": str(sample["selected_route"]),
            "synthetic_id": str(sample["synthetic_id"]),
            "synthetic_label": int(synthetic_labels[index]),
            "nearest_training_image_id": real_ids[best],
            "nearest_training_label": int(real_labels[best]),
            "nearest_training_patient_group": real_groups[best],
            "nearest_training_partition": real_parts[best],
            "feature_cosine_similarity": float(best_scores[index]),
            "ssim_to_nearest_feature_neighbor": float(ssim_values[index]),
            "nearest_same_class_image_id": real_ids[same_class],
            "same_class_feature_cosine_similarity": float(same_class_scores[index]),
            "minimum_training_dhash_hamming": int(min_dhash[index]),
            "dhash_near_duplicate_le_2": bool(min_dhash[index] <= 2),
            "exact_canonical_pixel_match": synthetic_hashes[index] in real_hash_set,
            "nearest_dhash_training_image_id": real_ids[hash_neighbor],
        })

    per_sample = pd.DataFrame(rows)
    reference_n = len(reference)
    reference_groups_n = int(reference["patient_group"].nunique())

    def summarize(method: str, seed: str, label: str, group: pd.DataFrame) -> dict:
        cosine = group["feature_cosine_similarity"].astype(float)
        ssim = group["ssim_to_nearest_feature_neighbor"].astype(float)
        ref_class_n = int(reference["label"].eq(int(label)).sum()) if label in {"0", "1"} else reference_n
        return {
            "generation_method": method,
            "seed": seed,
            "synthetic_label": label,
            "n_synthetic": int(len(group)),
            "n_reference_images": reference_n,
            "n_reference_patient_groups": reference_groups_n,
            "n_reference_images_same_class": ref_class_n,
            "mean_feature_cosine_similarity": float(cosine.mean()),
            "median_feature_cosine_similarity": float(cosine.median()),
            "q1_feature_cosine_similarity": float(cosine.quantile(0.25)),
            "q3_feature_cosine_similarity": float(cosine.quantile(0.75)),
            "max_feature_cosine_similarity": float(cosine.max()),
            "mean_ssim_to_nearest_feature_neighbor": float(ssim.mean()),
            "median_ssim_to_nearest_feature_neighbor": float(ssim.median()),
            "q1_ssim_to_nearest_feature_neighbor": float(ssim.quantile(0.25)),
            "q3_ssim_to_nearest_feature_neighbor": float(ssim.quantile(0.75)),
            "max_ssim_to_nearest_feature_neighbor": float(ssim.max()),
            "n_dhash_near_duplicate_le_2": int(group["dhash_near_duplicate_le_2"].sum()),
            "n_exact_canonical_pixel_match": int(group["exact_canonical_pixel_match"].sum()),
        }

    summary_rows: list[dict] = []
    for (method, seed), route_group in per_sample.groupby(["generation_method", "seed"], sort=True):
        summary_rows.append(summarize(str(method), str(seed), "all", route_group))
        for label, class_group in route_group.groupby("synthetic_label", sort=True):
            summary_rows.append(summarize(str(method), str(seed), str(int(label)), class_group))
    for method, method_group in per_sample.groupby("generation_method", sort=True):
        summary_rows.append(summarize(str(method), "all", "all", method_group))
        for label, class_group in method_group.groupby("synthetic_label", sort=True):
            summary_rows.append(summarize(str(method), "all", str(int(label)), class_group))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_sample.to_csv(args.output_dir / "per_sample_neighbors.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "summary.csv", index=False)
    metadata = {
        "status": "complete",
        "reference_scope": "all real images in the model-training cohort; generator-fit and checkpoint-monitor partitions only",
        "reference_images": int(reference_n),
        "reference_patient_groups": reference_groups_n,
        "reference_images_by_partition": {str(k): int(v) for k, v in partition_counts.items()},
        "patient_group_crossings_between_fit_and_monitor": 0,
        "reference_images_by_label": {str(k): int(v) for k, v in reference["label"].value_counts().sort_index().items()},
        "synthetic_images_by_method_seed": {
            f"{method}_{seed}": int(count) for (method, seed), count in route_counts.items()
        },
        "feature_encoder": "ImageNet ResNet50 IMAGENET1K_V1, L2-normalized 2048-dimensional penultimate features",
        "similarity_metric": "cosine similarity; higher values indicate closer features",
        "ssim_reference": "each synthetic image's nearest real training image by ResNet50 feature cosine similarity; 11x11 uniform local windows with valid-region averaging",
        "exact_match_definition": "SHA-256 of EXIF-corrected, aspect-preserving reflect-letterboxed RGB pixels at the analysis size",
        "near_duplicate_screen": "minimum 64-bit difference-hash Hamming distance <=2 against the real training reference",
        "analysis_image_size": int(args.image_size),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "weights_sha256": sha256(args.weights),
        "training_reference_manifest_sha256": sha256(args.training_reference),
        "synthetic_sample_manifest_sha256": sha256(args.samples),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
