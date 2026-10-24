#!/usr/bin/env python3
"""Summarize selected routes without reopening model selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (17, 42, 2024, 2025, 3407)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--selection-lock", type=Path, default=None)
    parser.add_argument("--test-split", choices=["test1", "source_holdout"], default="test1")
    args = parser.parse_args()
    root = args.output_root.resolve()
    prefix = "retrospective" if args.test_split == "test1" else "temporal"
    lock_path = args.selection_lock.resolve() if args.selection_lock else root / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selected_routes = set(lock["selected_routes"])
    summary_rows = []
    validation_rows = []
    test_rows = []
    manifest_by_route = {}
    for route in sorted(selected_routes):
        for seed in SEEDS:
            route_dir = root / "full_highres_matrix" / f"seed_{seed}" / route
            metrics_path = route_dir / "metrics_operating_points.csv"
            prediction_path = route_dir / "predictions.csv"
            if not metrics_path.is_file() or not prediction_path.is_file():
                raise FileNotFoundError(f"missing locked test artifacts: {route_dir}")
            manifest = json.loads((route_dir / "run_manifest.json").read_text(encoding="utf-8"))
            manifest_by_route[route] = manifest
            metrics = pd.read_csv(metrics_path)
            test = metrics[metrics["split"].astype(str).eq(args.test_split)].copy()
            if test.empty:
                raise ValueError(f"Missing {args.test_split} results: {route_dir}")
            test["route"] = route
            test["seed"] = seed
            summary_rows.append(test)
            predictions = pd.read_csv(prediction_path, dtype={"image_id": str})
            validation_rows.append(predictions[predictions["split"].astype(str).eq("val")].assign(route=route, seed=seed))
            test_rows.append(predictions[predictions["split"].astype(str).eq(args.test_split)].assign(route=route, seed=seed))
    metrics_frame = pd.concat(summary_rows, ignore_index=True)
    metrics_frame.to_csv(root / f"{prefix}_multiseed_test_metrics_by_seed.csv", index=False)
    grouped = (
        metrics_frame.groupby(["route", "operating_point"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std"),
            mean_f1=("f1", "mean"), sd_f1=("f1", "std"),
            mean_auroc=("auroc", "mean"), sd_auroc=("auroc", "std"),
            mean_auprc=("auprc", "mean"), sd_auprc=("auprc", "std"),
            mean_sensitivity=("sensitivity", "mean"), sd_sensitivity=("sensitivity", "std"),
            mean_specificity=("specificity", "mean"), sd_specificity=("specificity", "std"),
            mean_fnr=("fnr", "mean"), sd_fnr=("fnr", "std"),
        )
        .reset_index()
    )
    grouped["split"] = args.test_split
    grouped.to_csv(root / f"{prefix}_multiseed_summary.csv", index=False)

    ensemble_rows = []
    validation_ensemble_rows = []
    ensemble_metrics = []
    for route in sorted(selected_routes):
        val = pd.concat([row for row in validation_rows if row["route"].iloc[0] == route], ignore_index=True)
        test = pd.concat([row for row in test_rows if row["route"].iloc[0] == route], ignore_index=True)
        route_manifest = manifest_by_route[route]
        for part in (val, test):
            if part.groupby("image_id")["seed"].nunique().ne(len(SEEDS)).any() or part.duplicated(["image_id", "seed"]).any():
                raise ValueError(f"Incomplete or repeated seed predictions for {route}")
            if part.groupby("image_id")[["label", "patient_group"]].nunique().gt(1).any().any():
                raise ValueError(f"Seed predictions disagree on label/patient for {route}")
        val_mean = val.groupby("image_id", as_index=False).agg(
            label=("label", "first"), prob_malignant=("prob_malignant", "mean"), patient_group=("patient_group", "first")
        )
        test_mean = test.groupby("image_id", as_index=False).agg(
            label=("label", "first"), prob_malignant=("prob_malignant", "mean"), patient_group=("patient_group", "first")
        )
        from reproduce_gan_resnet50_xgb_patient224 import metrics, thresholds

        selected_thresholds = thresholds(val_mean["label"].to_numpy(), val_mean["prob_malignant"].to_numpy())
        for point, threshold in selected_thresholds.items():
            row = metrics(
                test_mean["label"].to_numpy(), test_mean["prob_malignant"].to_numpy(),
                float(threshold), route, 0, args.test_split + "_ensemble",
                float(route_manifest.get("ratio", route_manifest.get("requested_synthetic_ratio", 0.0))),
                224,
                bool(route_manifest.get("lidsentry", False)),
                str(route_manifest.get("strategy", "unknown")),
            )
            row.update({
                "route": route,
                "operating_point": point,
                "n_seed_probabilities": len(SEEDS),
                "generation_method": route_manifest.get("generation_method"),
                "backbone": route_manifest.get("backbone"),
                "classifier_head": route_manifest.get("head"),
                "strong_conventional": bool(route_manifest.get("strong_conventional", False)),
            })
            ensemble_metrics.append(row)
        test_mean["route"] = route
        test_mean["split"] = args.test_split + "_ensemble"
        test_mean["n_seed_probabilities"] = len(SEEDS)
        ensemble_rows.append(test_mean)
        val_mean["route"] = route
        val_mean["split"] = "val_ensemble"
        val_mean["n_seed_probabilities"] = len(SEEDS)
        validation_ensemble_rows.append(val_mean)
    pd.concat(ensemble_rows, ignore_index=True).to_csv(root / f"{prefix}_ensemble_predictions.csv", index=False)
    pd.concat(validation_ensemble_rows, ignore_index=True).to_csv(
        root / f"{prefix}_ensemble_validation_predictions.csv", index=False
    )
    pd.DataFrame(ensemble_metrics).to_csv(root / f"{prefix}_ensemble_metrics.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
