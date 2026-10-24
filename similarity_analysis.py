#!/usr/bin/env python3
"""Compare synthetic images with the real images used to fit each generator.

The script deliberately filters the supplied partition manifest to rows marked
``fit``. Monitoring, validation, and test images are not eligible references.
It writes per-image matches and an aggregate summary to the requested output
directory; it does not copy or export image pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


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
    resized = image.resize(target, Image.Resampling.LANCZOS)
    left = (size - target[0]) // 2
    top = (size - target[1]) // 2
    right = size - target[0] - left
    bottom = size - target[1] - top
    array = np.asarray(resized, dtype=np.uint8)
    padded = np.pad(array, ((top, bottom), (left, right), (0, 0)), mode="reflect")
    return np.asarray(padded, dtype=np.uint8)


def difference_hash(array: np.ndarray) -> int:
    gray = Image.fromarray(array, mode="RGB").convert("L").resize((9, 8), Image.Resampling.LANCZOS)
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


def load_features(paths: list[str], model, torch, size: int, batch_size: int) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    features: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = [letterbox(path, size) for path in paths[start:start + batch_size]]
            array = np.stack(images).transpose(0, 3, 1, 2).copy()
            values = torch.from_numpy(array).to(dtype=torch.float32).div_(255.0)
            values.sub_(mean).div_(std)
            batch = model(values)
            batch = torch.nn.functional.normalize(batch, p=2, dim=1)
            features.append(batch.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(features, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path,
                        help="CSV with generation_method, synthetic_id, synthetic_label, synthetic_path")
    parser.add_argument("--training-partition", required=True, type=Path,
                        help="Generator partition CSV containing fit and monitor rows")
    parser.add_argument("--comparison-partition", type=Path,
                        help="Optional second generator partition to verify identical fit membership")
    parser.add_argument("--weights", required=True, type=Path,
                        help="Cached torchvision ResNet18 ImageNet V1 state dictionary")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.image_size < 32 or args.batch_size < 1 or args.threads < 1:
        parser.error("image size must be >=32; batch size and threads must be positive")
    if not args.weights.is_file():
        parser.error(f"ResNet18 weights are not present at {args.weights}; provide a local cached file")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    import torch
    from torchvision import models

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(min(args.threads, 2))
    except RuntimeError:
        pass
    # CPU-only post-processing: no CUDA context or training process is started.
    model = models.resnet18(weights=None)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.fc = torch.nn.Identity()
    model.eval()

    samples = pd.read_csv(args.samples)
    required_samples = {"generation_method", "synthetic_id", "synthetic_label", "synthetic_path"}
    missing = sorted(required_samples - set(samples.columns))
    if missing:
        parser.error(f"sample manifest is missing columns: {missing}")
    samples["synthetic_label"] = pd.to_numeric(samples["synthetic_label"], errors="raise").astype(int)
    samples["synthetic_path"] = samples["synthetic_path"].astype(str)
    if samples["synthetic_id"].duplicated().any() or samples["synthetic_path"].duplicated().any():
        parser.error("sample IDs and image paths must be unique")
    if not set(samples["synthetic_label"].unique()).issubset({0, 1}):
        parser.error("labels must be encoded as 0 or 1")

    partition = pd.read_csv(args.training_partition, dtype={"patient_group": str, "image_id": str})
    required_partition = {"generator_partition", "patient_group", "image_id", "label", "absolute_path"}
    missing = sorted(required_partition - set(partition.columns))
    if missing:
        parser.error(f"training partition is missing columns: {missing}")
    fit = partition.loc[partition["generator_partition"].astype(str).eq("fit")].copy()
    monitor = partition.loc[partition["generator_partition"].astype(str).eq("monitor")].copy()
    if fit.empty or fit["image_id"].duplicated().any():
        parser.error("fit partition is empty or contains duplicate image IDs")
    if set(fit["patient_group"].astype(str)) & set(monitor["patient_group"].astype(str)):
        parser.error("patient groups cross the fit and monitor partitions")
    if not set(fit["label"].astype(int).unique()).issubset({0, 1}):
        parser.error("fit labels must be encoded as 0 or 1")
    if args.comparison_partition is not None:
        other = pd.read_csv(args.comparison_partition, dtype={"patient_group": str, "image_id": str})
        if not required_partition.issubset(other.columns):
            parser.error("comparison partition is missing required manifest columns")
        other_fit = other.loc[other["generator_partition"].astype(str).eq("fit")].copy()
        membership = ["image_id", "patient_group", "label"]
        left = fit[membership].assign(label=fit["label"].astype(int)).sort_values("image_id").reset_index(drop=True)
        right = other_fit[membership].assign(label=other_fit["label"].astype(int)).sort_values("image_id").reset_index(drop=True)
        if not left.equals(right):
            parser.error("the two generators do not have identical fit image, patient-group, and label membership")

    real_paths = fit["absolute_path"].astype(str).tolist()
    synthetic_paths = samples["synthetic_path"].tolist()
    all_paths = real_paths + synthetic_paths
    missing_paths = [path for path in all_paths if not Path(path).is_file()]
    if missing_paths:
        parser.error(f"{len(missing_paths)} referenced images are missing; first: {missing_paths[0]}")

    real_features = load_features(real_paths, model, torch, args.image_size, args.batch_size)
    synthetic_features = load_features(synthetic_paths, model, torch, args.image_size, args.batch_size)
    real_features = real_features / np.maximum(np.linalg.norm(real_features, axis=1, keepdims=True), 1e-12)
    synthetic_features = synthetic_features / np.maximum(np.linalg.norm(synthetic_features, axis=1, keepdims=True), 1e-12)
    similarity = synthetic_features @ real_features.T
    real_labels = fit["label"].astype(int).to_numpy()
    synthetic_labels = samples["synthetic_label"].to_numpy()
    real_ids = fit["image_id"].astype(str).to_numpy()
    real_groups = fit["patient_group"].astype(str).to_numpy()

    real_hashes: list[str] = []
    real_dhashes: list[int] = []
    for path in real_paths:
        array = letterbox(path, args.image_size)
        real_hashes.append(hashlib.sha256(array.tobytes()).hexdigest())
        real_dhashes.append(difference_hash(array))
    real_dhash_array = np.asarray(real_dhashes, dtype=np.uint64)
    real_hash_set = set(real_hashes)

    rows: list[dict] = []
    for index, sample in samples.iterrows():
        best = int(np.argmax(similarity[index]))
        same_class_indices = np.flatnonzero(real_labels == synthetic_labels[index])
        if not len(same_class_indices):
            parser.error(f"fit partition has no real examples for class {synthetic_labels[index]}")
        same_class_best = int(same_class_indices[np.argmax(similarity[index, same_class_indices])])
        synthetic_array = letterbox(sample["synthetic_path"], args.image_size)
        synthetic_gray = np.asarray(Image.fromarray(synthetic_array).convert("L"), dtype=np.uint8)
        neighbor_array = letterbox(real_paths[best], args.image_size)
        neighbor_gray = np.asarray(Image.fromarray(neighbor_array).convert("L"), dtype=np.uint8)
        ssim = structural_similarity(synthetic_gray, neighbor_gray)
        sample_hash = hashlib.sha256(synthetic_array.tobytes()).hexdigest()
        sample_dhash = difference_hash(synthetic_array)
        distances = np.fromiter((int(sample_dhash ^ int(value)).bit_count() for value in real_dhash_array),
                                dtype=np.int16, count=len(real_dhash_array))
        min_dhash_index = int(np.argmin(distances))
        rows.append({
            "generation_method": str(sample["generation_method"]),
            "synthetic_id": str(sample["synthetic_id"]),
            "synthetic_label": int(synthetic_labels[index]),
            "nearest_fit_image_id": real_ids[best],
            "nearest_fit_label": int(real_labels[best]),
            "nearest_fit_patient_group": real_groups[best],
            "feature_cosine_similarity": float(similarity[index, best]),
            "ssim_to_nearest_feature_neighbor": ssim,
            "nearest_same_class_image_id": real_ids[same_class_best],
            "same_class_feature_cosine_similarity": float(similarity[index, same_class_best]),
            "minimum_fit_dhash_hamming": int(distances[min_dhash_index]),
            "dhash_near_duplicate_le_2": bool(distances[min_dhash_index] <= 2),
            "exact_canonical_pixel_match": sample_hash in real_hash_set,
            "nearest_dhash_fit_image_id": real_ids[min_dhash_index],
        })

    per_sample = pd.DataFrame(rows)
    def summarize(method: str, label: str, group: pd.DataFrame) -> dict:
        return {
            "generation_method": method,
            "synthetic_label": label,
            "n_synthetic": int(len(group)),
            "n_fit_real_reference_images": int(len(fit)),
            "n_fit_real_reference_patient_groups": int(fit["patient_group"].nunique()),
            "mean_feature_cosine_similarity": float(group["feature_cosine_similarity"].mean()),
            "median_feature_cosine_similarity": float(group["feature_cosine_similarity"].median()),
            "max_feature_cosine_similarity": float(group["feature_cosine_similarity"].max()),
            "mean_ssim_to_nearest_feature_neighbor": float(group["ssim_to_nearest_feature_neighbor"].mean()),
            "median_ssim_to_nearest_feature_neighbor": float(group["ssim_to_nearest_feature_neighbor"].median()),
            "max_ssim_to_nearest_feature_neighbor": float(group["ssim_to_nearest_feature_neighbor"].max()),
            "n_dhash_near_duplicate_le_2": int(group["dhash_near_duplicate_le_2"].sum()),
            "n_exact_canonical_pixel_match": int(group["exact_canonical_pixel_match"].sum()),
        }

    summary_rows: list[dict] = []
    for method, method_group in per_sample.groupby("generation_method", sort=True):
        summary_rows.append(summarize(str(method), "all", method_group))
        for label, class_group in method_group.groupby("synthetic_label", sort=True):
            summary_rows.append(summarize(str(method), str(int(label)), class_group))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_sample.to_csv(args.output_dir / "per_sample_neighbors.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "summary.csv", index=False)
    fit_ids_sha256 = hashlib.sha256("\n".join(sorted(fit["image_id"].astype(str))).encode("utf-8")).hexdigest()
    fit_groups_sha256 = hashlib.sha256("\n".join(sorted(fit["patient_group"].astype(str).unique())).encode("utf-8")).hexdigest()
    metadata = {
        "status": "complete",
        "reference_partition": "generator_partition=fit only",
        "monitor_partition_used_as_reference": False,
        "patient_group_crossings_between_fit_and_monitor": 0,
        "real_fit_images": int(len(fit)),
        "real_fit_patient_groups": int(fit["patient_group"].nunique()),
        "fit_image_ids_sha256": fit_ids_sha256,
        "fit_patient_group_ids_sha256": fit_groups_sha256,
        "real_fit_images_by_label": {str(k): int(v) for k, v in fit["label"].astype(int).value_counts().sort_index().items()},
        "synthetic_images_by_method": {str(k): int(v) for k, v in samples.groupby("generation_method").size().items()},
        "feature_encoder": "ImageNet ResNet18 IMAGENET1K_V1, normalized 512-dimensional penultimate features",
        "similarity_metric": "cosine similarity; higher values indicate closer features",
        "ssim_reference": "nearest image in the full generator-fit partition by feature cosine similarity; 11x11 uniform window, 8-bit luminance range",
        "exact_match_definition": "SHA-256 of EXIF-corrected, reflect-letterboxed RGB pixels at the analysis size",
        "near_duplicate_screen": "minimum difference-hash Hamming distance <=2",
        "analysis_image_size": int(args.image_size),
        "device": "CPU",
        "weights_sha256": sha256(args.weights),
        "training_partition_sha256": sha256(args.training_partition),
        "sample_manifest_sha256": sha256(args.samples),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
