#!/usr/bin/env python3
"""Select study routes from validation artifacts only.

The script deliberately never reads ``metrics_operating_points.csv`` or
``predictions.csv``.  It consumes only validation-stage artifacts and writes a
machine-readable lock before held-out evaluation is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (17, 42, 2024, 2025, 3407)
HEAD_COMPLEXITY = {"cnn": 0, "xgboost": 1, "random_forest": 2, "svm": 3, "naive_bayes": 4}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_validation_rows(root: Path, expected_seeds: set[int]) -> pd.DataFrame:
    matrix_root = root / "full_highres_matrix"
    rows: list[dict] = []
    for seed in sorted(expected_seeds):
        seed_root = matrix_root / f"seed_{seed}"
        if not seed_root.is_dir():
            raise FileNotFoundError(f"missing seed directory: {seed_root}")
        # A seed directory also contains private representation caches such as
        # ``_representations``.  Only directories with a route manifest are
        # validation routes; cache directories must not be treated as failures.
        route_dirs = sorted(
            path
            for path in seed_root.iterdir()
            if path.is_dir() and (path / "run_manifest.json").is_file()
        )
        for route_dir in route_dirs:
            manifest_path = route_dir / "run_manifest.json"
            metrics_path = route_dir / "validation_metrics.csv"
            marker_path = route_dir / "validation_complete.json"
            if not manifest_path.is_file() or not metrics_path.is_file() or not marker_path.is_file():
                raise FileNotFoundError(f"missing validation-stage artifact for {route_dir}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("status") != "validation_complete" or manifest.get("stage") != "validation":
                raise RuntimeError(f"route is not validation-only: {route_dir}")
            metrics = pd.read_csv(metrics_path)
            metrics = metrics[metrics["split"].astype(str).eq("val")].copy()
            if metrics.empty:
                raise RuntimeError(f"validation metrics are empty: {metrics_path}")
            first = metrics.iloc[0].to_dict()
            rows.append({
                "route": str(manifest.get("route", route_dir.name)),
                "seed": int(manifest.get("seed", seed)),
                "generation_method": str(manifest.get("generation_method")),
                "ratio": float(manifest.get("ratio", manifest.get("requested_synthetic_ratio", 0))),
                "strategy": str(manifest.get("strategy")),
                "head": str(manifest.get("head")),
                "strong_conventional": bool(manifest.get("strong_conventional", False)),
                "val_auroc": float(first["auroc"]),
                "val_auprc": float(first["auprc"]),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no validation artifacts were found")
    observed = set(frame["seed"].astype(int))
    if observed != expected_seeds:
        raise RuntimeError(f"validation seeds {sorted(observed)} differ from expected {sorted(expected_seeds)}")
    route_seed_counts = frame.groupby("route")["seed"].nunique()
    incomplete = route_seed_counts[route_seed_counts != len(expected_seeds)]
    if not incomplete.empty:
        raise RuntimeError(f"routes missing validation seeds: {incomplete.to_dict()}")
    return frame


def choose(candidates: pd.DataFrame, expected_seeds: set[int]) -> tuple[pd.Series, pd.DataFrame]:
    summary = (
        candidates.groupby(["route", "generation_method", "ratio", "strategy", "head", "strong_conventional"], dropna=False)
        .agg(
            mean_validation_auroc=("val_auroc", "mean"),
            mean_validation_auprc=("val_auprc", "mean"),
            sd_validation_auroc=("val_auroc", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    if set(summary["n_seeds"].astype(int)) != {len(expected_seeds)}:
        raise RuntimeError("selection candidates are not complete across the declared seeds")
    summary["head_complexity"] = summary["head"].map(HEAD_COMPLEXITY).fillna(99).astype(int)
    ordered = summary.sort_values(
        ["mean_validation_auroc", "mean_validation_auprc", "ratio", "head_complexity", "route"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return ordered.iloc[0], ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--expected-seeds", default=','.join(str(value) for value in SEEDS))
    args = parser.parse_args()
    root = args.output_root.resolve()
    expected_seeds = {int(value) for value in args.expected_seeds.split(",") if value.strip()}
    if expected_seeds != set(SEEDS):
        raise ValueError(f"Formal selection requires seeds {SEEDS}")
    rows = collect_validation_rows(root, expected_seeds)
    selected_rows: list[pd.Series] = []
    candidate_frames: list[pd.DataFrame] = []
    # Primary generation-family selection is restricted to the predeclared
    # original-ratio routes.  The n=4 balanced route is a class-balance
    # sensitivity analysis and must not become the primary winner because its
    # neighboring n=3/n=5 routes do not exist under that strategy.
    gan = rows[
        (rows["generation_method"] == "gan")
        & ~rows["strong_conventional"]
        & rows["strategy"].eq("original_ratio")
    ]
    diffusion = rows[
        (rows["generation_method"] == "diffusion")
        & ~rows["strong_conventional"]
        & rows["strategy"].eq("original_ratio")
    ]
    # The strong-baseline family is deliberately real-only.  The matched
    # strong GAN/diffusion routes are controls, not candidates that can turn
    # the conventional baseline into another synthetic winner.
    strong = rows[rows["strong_conventional"] & rows["generation_method"].eq("real_only")]
    for name, candidates in [("gan", gan), ("diffusion", diffusion), ("strong_baseline", strong)]:
        if candidates.empty:
            raise RuntimeError(f"no validation candidates for {name}")
        selected, summary = choose(candidates, expected_seeds)
        selected = selected.copy()
        selected["selection_family"] = name
        selected_rows.append(selected)
        summary["selection_family"] = name
        candidate_frames.append(summary)
    selected_frame = pd.DataFrame(selected_rows)
    candidate_frame = pd.concat(candidate_frames, ignore_index=True)
    bridge_candidates = selected_frame[selected_frame["selection_family"].isin(["gan", "diffusion"])].copy()
    bridge_candidates = bridge_candidates.sort_values(
        ["mean_validation_auroc", "mean_validation_auprc", "ratio", "head_complexity", "route"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    if bridge_candidates.empty:
        raise RuntimeError("No GAN or diffusion route is available for the downstream comparison")
    selected_bridge = bridge_candidates.iloc[0]
    # Predeclare the neighboring-ratio and class-balance comparisons before
    # any test artifact is opened.  They are confirmatory stability analyses,
    # not routes selected from test performance.
    selected_routes = set(selected_frame["route"].astype(str))
    selected_routes.update(rows.loc[rows["strong_conventional"], "route"].astype(str).unique().tolist())
    for family in ["gan", "diffusion"]:
        family_choice = selected_frame.loc[selected_frame["selection_family"].eq(family)].iloc[0]
        candidates = rows[
            (rows["generation_method"] == family)
            & ~rows["strong_conventional"]
            & rows["head"].eq(family_choice["head"])
            & rows["strategy"].eq(family_choice["strategy"])
            & rows["ratio"].isin([3.0, 4.0, 5.0])
        ]
        selected_routes.update(candidates["route"].astype(str).unique().tolist())
        balanced = rows[
            (rows["generation_method"] == family)
            & ~rows["strong_conventional"]
            & rows["head"].eq(family_choice["head"])
            & rows["ratio"].eq(4.0)
            & rows["strategy"].eq("balanced")
        ]
        selected_routes.update(balanced["route"].astype(str).unique().tolist())
        real_match = rows[
            (rows["generation_method"] == "real_only")
            & ~rows["strong_conventional"]
            & rows["head"].eq(family_choice["head"])
            & rows["ratio"].eq(0.0)
            & rows["strategy"].eq("original_ratio")
        ]
        selected_routes.update(real_match["route"].astype(str).unique().tolist())
    selected_routes = sorted(selected_routes)
    lock = {
        "status": "locked",
        "protocol": "validation_only_route_selection",
        "selection_family_routes": {
            row["selection_family"]: row["route"] for _, row in selected_frame.iterrows()
        },
        "selected_bridge_route": str(selected_bridge["route"]),
        "selected_bridge_generation_method": str(selected_bridge["generation_method"]),
        "selected_bridge_ratio": float(selected_bridge["ratio"]),
        "selected_bridge_strategy": str(selected_bridge["strategy"]),
        "selected_bridge_head": str(selected_bridge["head"]),
        "selected_routes": selected_routes,
        "selection_rule": {
            "primary": "mean validation AUROC across five seeds",
            "tie_1": "mean validation AUPRC",
            "tie_2": "lower synthetic ratio",
            "tie_3": "simpler downstream head",
        },
        "expected_seeds": sorted(expected_seeds),
        "test_metrics_used": False,
        "test_predictions_read": False,
        "validation_artifact": "validation_metrics.csv",
        "split_sha256": sha256_file(Path(os.environ["LOCKED_SPLIT"])) if os.environ.get("MANUAL_EYE_DATASET_ROOT")
        else sha256_file(Path(__file__).with_name("locked_split.csv")),
        "manifest_sha256": sha256_file(Path(os.environ["MANIFEST_PATH"])) if os.environ.get("MANUAL_EYE_DATASET_ROOT")
        else sha256_file(Path(__file__).with_name("manifest_raw.csv")),
        "manual_dataset_root": os.environ.get("MANUAL_EYE_DATASET_ROOT"),
        "code_sha256": sha256_file(Path(__file__).with_name("full_highres_matrix.py")),
        "selection_script_sha256": sha256_file(Path(__file__).resolve()),
        "contract_sha256": sha256_file(Path(__file__).with_name(
            "manual_eye_training_contract.json"
            if os.environ.get("MANUAL_EYE_DATASET_ROOT")
            else "full_highres_matrix_contract.json"
        )),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
    }
    lock_path = root / "selection_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, default=str), encoding="utf-8")
    candidate_frame.to_csv(root / "selection_candidates.csv", index=False)
    print(json.dumps(lock, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
