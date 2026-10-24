#!/usr/bin/env python3
"""Assemble private training-image manifests for the final five-seed audit.

This script reads existing completed model routes and generator partition
manifests. It never copies image pixels or model checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SEEDS = (17, 42, 2024, 2025, 3407)
ROUTES = {
    "gan": ("gan_resnet50_svm_n1_original_ratio_nolidsentry_size224", 1, 1695, {0: 1145, 1: 550}),
    "diffusion": ("diffusion_resnet50_cnn_n3_original_ratio_nolidsentry_size224", 3, 5085, {0: 3435, 1: 1650}),
}
PARTITION_FIELDS = ("image_id", "patient_group", "label", "generator_partition", "absolute_path")
COHORT_FIELDS = ("image_id", "patient_group", "label", "absolute_path")


def read_partition(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"image_id": str, "patient_group": str, "absolute_path": str})
    missing = set(PARTITION_FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.loc[frame["generator_partition"].astype(str).isin(("fit", "monitor")), list(PARTITION_FIELDS)].copy()
    if frame[list(PARTITION_FIELDS)].isna().any().any():
        raise ValueError(f"Missing training-reference identifiers, labels, or paths in {path}")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    frame = frame.sort_values(["generator_partition", "image_id"]).reset_index(drop=True)
    if frame["image_id"].duplicated().any():
        raise ValueError(f"Duplicate training image IDs in {path}")
    if frame["absolute_path"].duplicated().any():
        raise ValueError(f"Duplicate training paths in {path}")
    if not set(frame["label"]).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {path}")
    fit_groups = set(frame.loc[frame["generator_partition"].eq("fit"), "patient_group"])
    monitor_groups = set(frame.loc[frame["generator_partition"].eq("monitor"), "patient_group"])
    if fit_groups & monitor_groups:
        raise ValueError(f"Patient groups cross fit and monitor partitions in {path}")
    return frame


def digest_membership(frame: pd.DataFrame) -> str:
    """Hash cohort identity independently of seed-specific fit/monitor assignment."""
    values = frame.loc[:, list(COHORT_FIELDS)].astype(str).sort_values("image_id").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def truth_mask(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", required=True, type=Path,
                        help="full_highres_matrix directory containing seed-specific route folders")
    parser.add_argument("--gan-partition-root", required=True, type=Path,
                        help="synthetic directory containing gan_training_partition.csv per seed")
    parser.add_argument("--diffusion-partition-root", required=True, type=Path,
                        help="synthetic_diffusion directory containing diffusion_training_partition.csv per seed")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="private working directory for manifests; do not publish these files")
    parser.add_argument("--expected-train-images", type=int, default=1695)
    args = parser.parse_args()

    reference_paths = [
        args.gan_partition_root / f"seed_{seed}" / "gan_training_partition.csv"
        for seed in SEEDS
    ] + [
        args.diffusion_partition_root / f"seed_{seed}" / "diffusion_training_partition.csv"
        for seed in SEEDS
    ]
    references = [read_partition(path) for path in reference_paths]
    first = references[0]
    if len(first) != args.expected_train_images:
        raise ValueError(f"Expected {args.expected_train_images} training references, found {len(first)}")
    reference_digest = digest_membership(first)
    if any(digest_membership(frame) != reference_digest for frame in references[1:]):
        raise ValueError("Training-cohort image, patient-group, label, or source-path membership differs across seeds or generators")

    partition_records = []
    reference_specifications = [
        (seed, "gan", args.gan_partition_root / f"seed_{seed}" / "gan_training_partition.csv")
        for seed in SEEDS
    ] + [
        (seed, "diffusion", args.diffusion_partition_root / f"seed_{seed}" / "diffusion_training_partition.csv")
        for seed in SEEDS
    ]
    for (seed, method, path), frame in zip(reference_specifications, references):
        counts = frame["generator_partition"].value_counts().to_dict()
        partition_records.append({
            "seed": seed,
            "generation_method": method,
            "fit_images": int(counts.get("fit", 0)),
            "checkpoint_monitor_images": int(counts.get("monitor", 0)),
            "training_cohort_images": int(len(frame)),
            "patient_groups": int(frame["patient_group"].nunique()),
            "source_manifest": str(path),
        })

    sample_frames: list[pd.DataFrame] = []
    route_records = []
    for seed in SEEDS:
        for method, (route, ratio, expected_n, expected_classes) in ROUTES.items():
            route_dir = args.matrix_root / f"seed_{seed}" / route
            complete_path = route_dir / "run_complete.json"
            manifest_path = route_dir / "training_manifest.csv"
            if not complete_path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"Selected model route is incomplete: {route_dir}")
            completion = json.loads(complete_path.read_text(encoding="utf-8"))
            if completion.get("status") != "complete" or int(completion.get("seed", -1)) != seed:
                raise ValueError(f"Completion marker does not match the selected route: {complete_path}")
            if completion.get("route") != route:
                raise ValueError(f"Route identity mismatch in {complete_path}")

            frame = pd.read_csv(
                manifest_path,
                usecols=["image_id", "absolute_path", "label", "synthetic", "requested_synthetic_ratio",
                         "class_generation_strategy", "generation_method"],
                dtype={"image_id": str, "absolute_path": str},
            )
            synthetic = frame.loc[truth_mask(frame["synthetic"])].copy()
            synthetic["label"] = pd.to_numeric(synthetic["label"], errors="raise").astype(int)
            class_counts = synthetic["label"].value_counts().sort_index().to_dict()
            if len(synthetic) != expected_n or class_counts != expected_classes:
                raise ValueError(
                    f"{route} seed {seed}: expected {expected_n} images / {expected_classes}, "
                    f"found {len(synthetic)} / {class_counts}"
                )
            if synthetic["absolute_path"].duplicated().any() or synthetic["image_id"].duplicated().any():
                raise ValueError(f"Duplicate synthetic image identifiers or paths in {manifest_path}")
            if not (pd.to_numeric(synthetic["requested_synthetic_ratio"], errors="raise") == ratio).all():
                raise ValueError(f"Unexpected synthetic ratio in {manifest_path}")
            if not synthetic["class_generation_strategy"].astype(str).eq("original_ratio").all():
                raise ValueError(f"Unexpected class-generation strategy in {manifest_path}")
            if not synthetic["generation_method"].astype(str).str.lower().eq(method).all():
                raise ValueError(f"Unexpected generator label in {manifest_path}")

            synthetic = synthetic.rename(columns={
                "image_id": "synthetic_id",
                "label": "synthetic_label",
            })
            synthetic["generation_method"] = method
            synthetic["seed"] = seed
            synthetic["selected_route"] = route
            sample_frames.append(synthetic[[
                "generation_method", "seed", "selected_route", "synthetic_id",
                "synthetic_label", "absolute_path",
            ]].rename(columns={"absolute_path": "synthetic_path"}))
            route_records.append({
                "seed": seed,
                "generation_method": method,
                "selected_route": route,
                "ratio": ratio,
                "n_synthetic": int(len(synthetic)),
                "class_counts": {str(k): int(v) for k, v in class_counts.items()},
                "completion_marker": str(complete_path),
            })

    samples = pd.concat(sample_frames, ignore_index=True)
    if samples["synthetic_path"].duplicated().any():
        raise ValueError("A synthetic image path is assigned to more than one selected training run")
    if len(samples) != sum(record["n_synthetic"] for record in route_records):
        raise AssertionError("Combined sample count does not match route records")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = first.loc[:, list(COHORT_FIELDS)].sort_values("image_id", kind="stable").reset_index(drop=True)
    reference["reference_scope"] = "training_cohort"
    reference = reference[["reference_scope", "image_id", "patient_group", "label", "absolute_path"]]
    reference_path = args.output_dir / "training_reference_1695.csv"
    sample_path = args.output_dir / "selected_synthetic_samples.csv"
    reference.to_csv(reference_path, index=False)
    samples.to_csv(sample_path, index=False)
    summary = {
        "status": "complete",
        "seeds": list(SEEDS),
        "training_reference_images": int(len(reference)),
        "training_reference_images_by_label": {
            str(k): int(v) for k, v in reference["label"].value_counts().sort_index().items()
        },
        "training_reference_patient_groups": int(reference["patient_group"].nunique()),
        "reference_membership_sha256": reference_digest,
        "source_fit_monitor_partition_counts": partition_records,
        "synthetic_samples_total": int(len(samples)),
        "selected_routes": route_records,
        "note": "Private manifests contain image paths and patient-group identifiers; keep on the approved remote.",
    }
    (args.output_dir / "input_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "training_reference_images": len(first),
        "synthetic_samples_total": len(samples),
        "samples_by_method_seed": {
            f"{method}_{seed}": int(count)
            for (method, seed), count in samples.groupby(["generation_method", "seed"]).size().items()
        },
        "source_fit_monitor_partition_counts": [
            {k: v for k, v in record.items() if k not in {"source_manifest"}}
            for record in partition_records
        ],
        "routes_completed": len(route_records),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
