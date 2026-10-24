#!/usr/bin/env python3
"""Compute paired patient-group bootstrap deltas for predeclared comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (17, 42, 2024, 2025, 3407)
METRICS = ("accuracy", "f1", "auroc", "auprc", "sensitivity", "specificity", "fnr")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--test-split", choices=["test1", "source_holdout"], default="test1")
    args = parser.parse_args()
    root = args.output_root.resolve()
    prefix = "retrospective" if args.test_split == "test1" else "temporal"
    lock = json.loads((root / "selection_lock.json").read_text(encoding="utf-8"))
    selected = set(lock["selected_routes"])
    manifest_rows = []
    for route in selected:
        path = root / "full_highres_matrix" / f"seed_{SEEDS[0]}" / route / "run_manifest.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            manifest_rows.append({"route": route, **payload})
    manifests = pd.DataFrame(manifest_rows)
    prediction = pd.read_csv(root / f"{prefix}_ensemble_predictions.csv", dtype={"image_id": str, "patient_group": str})
    ensemble_metrics = pd.read_csv(root / f"{prefix}_ensemble_metrics.csv")
    from reproduce_gan_resnet50_xgb_patient224 import metrics

    def route_for(**criteria):
        current = manifests.copy()
        for key, value in criteria.items():
            current = current[current[key].astype(str).eq(str(value))]
        return sorted(current["route"].astype(str))[0] if not current.empty else None

    pairs = []
    for family in ["gan", "diffusion"]:
        selected_route = lock["selection_family_routes"].get(family)
        if not selected_route:
            continue
        choice = manifests[manifests["route"].eq(selected_route)].iloc[0]
        matched_real = route_for(
            generation_method="real_only", head=choice["head"], strategy="original_ratio",
            strong_conventional=False, ratio=0.0,
        )
        if matched_real:
            pairs.append((f"{family}_selected_vs_real", matched_real, selected_route))
        for ratio in [3.0, 5.0]:
            left = route_for(generation_method=family, head=choice["head"], strategy=choice["strategy"], strong_conventional=False, ratio=ratio)
            right = route_for(generation_method=family, head=choice["head"], strategy=choice["strategy"], strong_conventional=False, ratio=4.0)
            if left and right:
                pairs.append((f"{family}_n4_vs_n{int(ratio)}", left, right))
        original = route_for(generation_method=family, head=choice["head"], strategy="original_ratio", strong_conventional=False, ratio=4.0)
        balanced = route_for(generation_method=family, head=choice["head"], strategy="balanced", strong_conventional=False, ratio=4.0)
        if original and balanced:
            pairs.append((f"{family}_n4_balanced_vs_original", original, balanced))
    gan = lock["selection_family_routes"].get("gan")
    diffusion = lock["selection_family_routes"].get("diffusion")
    if gan and diffusion:
        pairs.append(("selected_gan_vs_diffusion", diffusion, gan))

    rows = []
    ensemble_split_name = f"{args.test_split}_ensemble"
    for pair_index, (pair_name, left_route, right_route) in enumerate(pairs):
        left = prediction[prediction["route"].eq(left_route)].copy()
        right = prediction[prediction["route"].eq(right_route)].copy()
        merged = left.merge(right, on=["image_id", "patient_group", "label"], suffixes=("_left", "_right"), validate="one_to_one")
        if merged.empty or len(merged) != len(left) or len(merged) != len(right):
            raise ValueError(f"Unmatched paired test predictions: {pair_name}")
        clusters = [part for _, part in merged.groupby("patient_group", sort=True)]
        rng = np.random.default_rng(91001 + pair_index)
        thresholds = {}
        for route in [left_route, right_route]:
            locked = ensemble_metrics[
                ensemble_metrics["route"].astype(str).eq(route)
            ]
            thresholds[route] = {
                str(row["operating_point"]): float(row["threshold"])
                for _, row in locked.iterrows()
                if "threshold" in row and pd.notna(row["threshold"])
            }
            if not thresholds[route]:
                raise ValueError(f"Missing validation-selected ensemble thresholds: {route}")
        for point in ["youden", "sens90", "sens95", "spec90"]:
            left_threshold = float(thresholds[left_route][point])
            right_threshold = float(thresholds[right_route][point])
            left_observed = metrics(merged["label"], merged["prob_malignant_left"], left_threshold, left_route, 0, ensemble_split_name, 0, 224, False, "original_ratio")
            right_observed = metrics(merged["label"], merged["prob_malignant_right"], right_threshold, right_route, 0, ensemble_split_name, 0, 224, False, "original_ratio")
            draws = {metric: [] for metric in METRICS}
            for _ in range(args.n_bootstrap):
                indices = rng.integers(0, len(clusters), len(clusters))
                sampled = pd.concat([clusters[index] for index in indices], ignore_index=True)
                left_metrics = metrics(sampled["label"], sampled["prob_malignant_left"], left_threshold, left_route, 0, ensemble_split_name, 0, 224, False, "original_ratio")
                right_metrics = metrics(sampled["label"], sampled["prob_malignant_right"], right_threshold, right_route, 0, ensemble_split_name, 0, 224, False, "original_ratio")
                for metric in METRICS:
                    delta = float(right_metrics[metric]) - float(left_metrics[metric])
                    if np.isfinite(delta):
                        draws[metric].append(delta)
            for metric in METRICS:
                values = np.asarray(draws[metric], dtype=float)
                observed_delta = float(right_observed[metric]) - float(left_observed[metric])
                rows.append({
                    "split": args.test_split,
                    "pair": pair_name,
                    "route_left": left_route,
                    "route_right": right_route,
                    "operating_point": point,
                    "metric": metric,
                    "observed_delta_right_minus_left": observed_delta,
                    "ci95_low": float(np.percentile(values, 2.5)) if values.size else np.nan,
                    "ci95_high": float(np.percentile(values, 97.5)) if values.size else np.nan,
                    "bootstrap_replicates": args.n_bootstrap,
                    "valid_bootstrap_replicates": int(values.size),
                    "resampling_unit": "patient_group",
                    "matched_images": int(len(merged)),
                })
    result = pd.DataFrame(rows)
    result.to_csv(root / f"{prefix}_paired_patient_bootstrap.csv", index=False)
    result.to_csv(root / f"{prefix}_paired_deltas.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
