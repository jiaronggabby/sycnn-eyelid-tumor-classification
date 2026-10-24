#!/usr/bin/env python3
"""Independent validation and patient-cluster uncertainty for the full matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BUNDLE = Path(__file__).resolve().parent
os.environ.setdefault("RUN_STAGE", "full_matrix")
os.environ.setdefault("OUTPUT_ROOT", str(BUNDLE / "_validation_output"))
sys.path.insert(0, str(BUNDLE))
import reproduce_gan_resnet50_xgb_patient224 as base  # noqa: E402


METRICS = [
    "accuracy", "f1", "mcc", "auroc", "auprc", "brier", "ece",
    "sensitivity", "specificity", "fpr", "fnr", "ppv", "npv",
]
BASE_SPLITS = ["val", "test1"]


def compare_float(left, right, tolerance=1e-8) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    try:
        return bool(np.isclose(float(left), float(right), rtol=1e-6, atol=tolerance, equal_nan=True))
    except (TypeError, ValueError):
        return str(left) == str(right)


def locked_split_audit() -> tuple[pd.DataFrame, dict]:
    if os.environ.get("MANUAL_EYE_DATASET_ROOT"):
        from manual_eye_training import load_manual_data
        frame, audit, _ = load_manual_data(Path(os.environ["MANUAL_EYE_DATASET_ROOT"]), base)
        return frame, audit
    split_path = Path(os.environ.get("LOCKED_SPLIT", BUNDLE / "locked_split.csv"))
    manifest_path = Path(os.environ.get("MANIFEST_PATH", BUNDLE / "manifest_raw.csv"))
    package_manifest_path = BUNDLE / "PACKAGE_MANIFEST.json"
    if package_manifest_path.is_file():
        package_manifest = json.loads(package_manifest_path.read_text(encoding="utf-8"))
        expected_hashes = {
            "manifest_raw.csv": package_manifest.get("manifest_sha256"),
            "locked_split.csv": package_manifest.get("locked_split_sha256"),
        }
        observed_hashes = {
            "manifest_raw.csv": base.sha256_file(manifest_path),
            "locked_split.csv": base.sha256_file(split_path),
        }
        mismatches = {
            name: {"expected": expected, "observed": observed_hashes[name]}
            for name, expected in expected_hashes.items()
            if expected and expected != observed_hashes[name]
        }
        if mismatches:
            raise RuntimeError(
                "validation received an input manifest or split outside the "
                "frozen current cohort: "
                + json.dumps(mismatches, sort_keys=True)
            )
    locked = pd.read_csv(split_path)
    manifest = pd.read_csv(manifest_path, usecols=["image_id", "original_split", "filename", "md5", "dhash"])
    locked = locked.merge(manifest, on="image_id", how="left", validate="one_to_one")
    if locked["original_split"].isna().any():
        raise RuntimeError("locked split contains image IDs absent from manifest_raw.csv")
    locked["split"] = locked["split"].astype(str).str.lower().replace({"test": "test1"})
    contract_path = BUNDLE / "full_highres_matrix_contract.json"
    if contract_path.is_file():
        expected = json.loads(contract_path.read_text(encoding="utf-8")).get("data_split", {})
        expected_images = int(expected.get("images", len(locked)))
        expected_groups = int(expected.get("patient_groups", locked["patient_group"].nunique()))
        if len(locked) != expected_images or locked["patient_group"].nunique() != expected_groups:
            raise RuntimeError(
                "active locked_split.csv does not match the frozen cohort contract: "
                + json.dumps({
                    "expected_images": expected_images,
                    "observed_images": int(len(locked)),
                    "expected_patient_groups": expected_groups,
                    "observed_patient_groups": int(locked["patient_group"].nunique()),
                }, sort_keys=True)
            )
    patient_cross = int(locked.groupby("patient_group")["split"].nunique().gt(1).sum())
    exact_link_cross = base.count_cross_split_exact_links(locked)
    near_duplicate_cross = base.count_cross_split_near_duplicates(locked)
    mixed = int(locked.groupby("patient_group")["label"].nunique().gt(1).sum())
    source_flags = locked.assign(_is_original_test=locked["original_split"].astype(str).str.lower().eq("test"))
    source_groups = set(source_flags.loc[source_flags["_is_original_test"], "patient_group"])
    audit = {
        "group_field": "patient_group",
        "n_images": int(len(locked)),
        "n_patient_groups": int(locked["patient_group"].nunique()),
        "patient_groups_crossing_splits": patient_cross,
        "exact_duplicate_links_crossing_splits": exact_link_cross,
        "patient_group_definition": "authoritative manifest patient_group",
        "near_duplicate_links_crossing_splits": near_duplicate_cross,
        "mixed_label_patient_groups": mixed,
        "source_holdout_images": int(locked["patient_group"].isin(source_groups).sum()),
        "source_holdout_patient_groups": int(len(source_groups)),
        "split_counts": {"|".join(map(str, key)): int(value) for key, value in locked.groupby(["split", "label_name"]).size().items()},
    }
    return locked, audit


def recompute_route(route_dir: Path, locked: pd.DataFrame) -> tuple[list[dict], list[dict], list[str]]:
    failures: list[str] = []
    manifest_path = route_dir / "run_manifest.json"
    complete_path = route_dir / "run_complete.json"
    prediction_path = route_dir / "predictions.csv"
    metric_path = route_dir / "metrics_operating_points.csv"
    confusion_path = route_dir / "confusion_matrices.csv"
    threshold_path = route_dir / "thresholds.json"
    required = [manifest_path, complete_path, prediction_path, metric_path, confusion_path, threshold_path]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return [], [], [f"{route_dir.name}: missing required route artifact"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(prediction_path)
        reported = pd.read_csv(metric_path)
        reported_confusion = pd.read_csv(confusion_path)
    except Exception as exc:
        return [], [], [f"{route_dir.name}: cannot read artifacts: {exc}"]
    route = str(manifest.get("route", route_dir.name))
    if route != route_dir.name:
        failures.append(f"{route_dir.name}: manifest route does not match directory name ({route})")
    if complete.get("status") != "complete":
        failures.append(f"{route}: completion marker is not complete")
    if complete.get("configuration_fingerprint") != manifest.get("configuration_fingerprint"):
        failures.append(f"{route}: completion fingerprint differs from run manifest")
    if manifest.get("input_size") != 224 or manifest.get("generator_size") != 256:
        failures.append(f"{route}: non-locked resolution in manifest")
    if manifest.get("group_field") != "patient_group":
        failures.append(f"{route}: grouping field must be patient_group")
    if manifest.get("backbone") != "resnet50":
        failures.append(f"{route}: backbone is outside the formal paper protocols")
    if manifest.get("backbone_initialization") != "imagenet" or manifest.get("pretrained_weights_used") is not True:
        failures.append(f"{route}: backbone initialization is not ImageNet-pretrained")
    if not manifest.get("python_version") or not manifest.get("package_versions"):
        failures.append(f"{route}: runtime environment trace is missing")
    if float(manifest.get("ratio", manifest.get("requested_synthetic_ratio", 0))) > 0:
        if not manifest.get("generation_configuration_fingerprint"):
            failures.append(f"{route}: synthetic route is missing generator fingerprint")
        if int(manifest.get("synthetic_pool_image_count", 0)) <= 0:
            failures.append(f"{route}: synthetic route is missing pool image count")
    if manifest.get("feature_dimension") != 512:
        failures.append(f"{route}: expected 512-D downstream feature representation")
    representation_relative = manifest.get("representation_artifact_dir")
    if not representation_relative:
        failures.append(f"{route}: shared representation artifact path is missing")
    else:
        representation_dir = route_dir.parent / Path(str(representation_relative))
        representation_marker = representation_dir / "representation_complete.json"
        representation_checkpoint = representation_dir / "backbone_best_state.pt"
        representation_history = representation_dir / "history.csv"
        if not all(path.is_file() and path.stat().st_size > 0 for path in [
            representation_marker, representation_checkpoint, representation_history,
        ]):
            failures.append(f"{route}: shared representation artifacts are incomplete")
        else:
            try:
                representation_payload = json.loads(
                    representation_marker.read_text(encoding="utf-8")
                )
                if representation_payload.get("representation_fingerprint") != manifest.get("representation_fingerprint"):
                    failures.append(f"{route}: shared representation fingerprint disagrees with route manifest")
                checkpoint_hash = base.sha256_file(representation_checkpoint)
                if manifest.get("representation_checkpoint_sha256") != checkpoint_hash:
                    failures.append(f"{route}: shared representation checkpoint hash disagrees with route manifest")
            except Exception as exc:
                failures.append(f"{route}: cannot validate shared representation artifacts: {exc}")
    expected_head_dim = 512 + (
        1 if bool(manifest.get("lidsentry", False)) and str(manifest.get("head")) != "cnn" else 0
    )
    if manifest.get("head_feature_dimension") != expected_head_dim:
        failures.append(
            f"{route}: head feature dimension={manifest.get('head_feature_dimension')} "
            f"expected {expected_head_dim}"
        )
    expected_native_dim = 2048 if manifest.get("backbone") == "resnet50" else None
    if expected_native_dim is not None and manifest.get("native_feature_dimension") != expected_native_dim:
        failures.append(
            f"{route}: native feature dimension={manifest.get('native_feature_dimension')} "
            f"expected {expected_native_dim}"
        )
    required_prediction_columns = {"route", "seed", "split", "image_id", "label", "prob_malignant", "patient_group"}
    missing = required_prediction_columns.difference(predictions.columns)
    if missing:
        failures.append(f"{route}: predictions missing {sorted(missing)}")
        return [], [], failures
    if predictions.duplicated(["split", "image_id"]).any():
        failures.append(f"{route}: duplicate image IDs within a prediction split")
    required_confusion_columns = {"route", "seed", "split", "operating_point", "threshold", "tp", "fp", "tn", "fn"}
    missing_confusion = required_confusion_columns.difference(reported_confusion.columns)
    if missing_confusion:
        failures.append(f"{route}: confusion_matrices.csv missing {sorted(missing_confusion)}")
    locked_index = locked.set_index("image_id")
    source_flags = locked.assign(_is_original_test=locked["original_split"].astype(str).str.lower().eq("test"))
    source_groups = set(source_flags.loc[source_flags["_is_original_test"], "patient_group"])
    expected_source_ids = set(locked.loc[locked["patient_group"].isin(source_groups), "image_id"])
    if not set(predictions["image_id"]).issubset(set(locked_index.index)):
        failures.append(f"{route}: predictions contain IDs outside locked real-image manifest")
    else:
        expected_labels = locked_index.reindex(predictions["image_id"])["label"].to_numpy()
        if not np.array_equal(expected_labels.astype(int), predictions["label"].to_numpy(dtype=int)):
            failures.append(f"{route}: prediction labels disagree with locked manifest")
        expected_groups = locked_index.reindex(predictions["image_id"])["patient_group"].astype(str).to_numpy()
        if not np.array_equal(expected_groups, predictions["patient_group"].astype(str).to_numpy()):
            failures.append(f"{route}: prediction patient groups disagree with locked manifest")
        for split in ["val", "test1"]:
            predicted_ids = set(predictions.loc[predictions["split"].astype(str).eq(split), "image_id"])
            expected_ids = set(locked.loc[
                locked["split"].astype(str).eq(split)
                & ~locked["patient_group"].isin(source_groups),
                "image_id",
            ])
            if predicted_ids != expected_ids:
                failures.append(f"{route}: {split} prediction membership differs from locked split")
        predicted_source_ids = set(predictions.loc[predictions["split"].astype(str).eq("source_holdout"), "image_id"])
        if predicted_source_ids != expected_source_ids:
            failures.append(f"{route}: source_holdout prediction membership differs from locked source cohort")
        training_manifest_path = route_dir / "training_manifest.csv"
        if training_manifest_path.is_file():
            training_manifest = pd.read_csv(training_manifest_path, usecols=["image_id"])
            leaked_source_ids = set(training_manifest["image_id"].astype(str)).intersection(expected_source_ids)
            if leaked_source_ids:
                failures.append(f"{route}: source-holdout images appear in training_manifest ({len(leaked_source_ids)})")
    split_group_counts = predictions.groupby("patient_group")["split"].agg(lambda values: set(values.astype(str)))
    invalid_group_overlap = [groups for groups in split_group_counts.tolist() if len(groups - {"source_holdout"}) > 1]
    if invalid_group_overlap:
        failures.append(f"{route}: patient group crosses validation/test splits")
    reported_rows: list[dict] = []
    recomputed_rows: list[dict] = []
    expected_splits = BASE_SPLITS + (["source_holdout"] if expected_source_ids else [])
    for split in expected_splits:
        prediction_group = predictions[predictions["split"].astype(str).eq(split)]
        if prediction_group.empty:
            failures.append(f"{route}: missing prediction split {split}")
            continue
        for point, threshold in thresholds.items():
            current = base.metrics(
                prediction_group["label"].to_numpy(),
                prediction_group["prob_malignant"].to_numpy(),
                float(threshold),
                route,
                int(manifest.get("seed", 0)),
                split,
                float(manifest.get("ratio", manifest.get("requested_synthetic_ratio", 0))),
                224,
                bool(manifest.get("lidsentry", False)),
                str(manifest.get("strategy", "unknown")),
            )
            current.update({
                "operating_point": point,
                "generation_method": manifest.get("generation_method"),
                "backbone": manifest.get("backbone"),
                "classifier_head": manifest.get("head"),
            })
            recomputed_rows.append(current)
            candidate = reported[
                reported["split"].astype(str).eq(split)
                & reported["operating_point"].astype(str).eq(str(point))
            ]
            if len(candidate) != 1:
                failures.append(f"{route}: expected one reported row for {split}/{point}, found {len(candidate)}")
                continue
            reported_row = candidate.iloc[0].to_dict()
            reported_row.update({"route": route})
            reported_rows.append(reported_row)
            if not missing_confusion:
                confusion_candidate = reported_confusion[
                    reported_confusion["split"].astype(str).eq(split)
                    & reported_confusion["operating_point"].astype(str).eq(str(point))
                ]
                if len(confusion_candidate) != 1:
                    failures.append(f"{route}: expected one confusion row for {split}/{point}, found {len(confusion_candidate)}")
                else:
                    confusion_row = confusion_candidate.iloc[0]
                    for metric in ["tp", "fp", "tn", "fn"]:
                        if int(confusion_row[metric]) != int(current[metric]):
                            failures.append(
                                f"{route}: confusion matrix mismatch {split}/{point}/{metric}: "
                                f"reported={confusion_row[metric]} recomputed={current[metric]}"
                            )
            for metric in METRICS + ["tp", "fp", "tn", "fn"]:
                if metric not in reported.columns:
                    failures.append(f"{route}: metrics file missing {metric}")
                    break
                if not compare_float(reported_row.get(metric), current.get(metric)):
                    failures.append(f"{route}: metric mismatch {split}/{point}/{metric}: reported={reported_row.get(metric)} recomputed={current.get(metric)}")
    return recomputed_rows, reported_rows, failures


def patient_bootstrap(route_dir: Path, n_boot: int) -> pd.DataFrame:
    predictions = pd.read_csv(route_dir / "predictions.csv")
    thresholds = json.loads((route_dir / "thresholds.json").read_text(encoding="utf-8"))
    manifest = json.loads((route_dir / "run_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for split in ["test1", "source_holdout"]:
        group = predictions[predictions["split"].astype(str).eq(split)].copy()
        if group.empty:
            continue
        clusters = [part for _, part in group.groupby("patient_group")]
        rng = np.random.default_rng(int(manifest.get("seed", 0)) + 9001)
        for point, threshold in thresholds.items():
            values = []
            for _ in range(n_boot):
                sampled = pd.concat([clusters[index] for index in rng.integers(0, len(clusters), len(clusters))], ignore_index=True)
                value = base.metrics(sampled["label"], sampled["prob_malignant"], float(threshold), manifest.get("route", route_dir.name), int(manifest.get("seed", 0)), split, float(manifest.get("ratio", 0)), 224, bool(manifest.get("lidsentry", False)), str(manifest.get("strategy", "unknown")))
                values.append(value)
            for metric in METRICS:
                array = np.asarray([item[metric] for item in values], dtype=float)
                finite = array[np.isfinite(array)]
                observed = base.metrics(group["label"], group["prob_malignant"], float(threshold), manifest.get("route", route_dir.name), int(manifest.get("seed", 0)), split, float(manifest.get("ratio", 0)), 224, bool(manifest.get("lidsentry", False)), str(manifest.get("strategy", "unknown")))[metric]
                rows.append({
                    "route": manifest.get("route", route_dir.name),
                    "seed": manifest.get("seed"),
                    "split": split,
                    "operating_point": point,
                    "metric": metric,
                    "observed_estimate": observed,
                    "ci95_low": float(np.nanpercentile(finite, 2.5)) if len(finite) else np.nan,
                    "ci95_high": float(np.nanpercentile(finite, 97.5)) if len(finite) else np.nan,
                    "valid_bootstrap_replicates": int(len(finite)),
                    "resampling_unit": "patient_group",
                })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--expected-seeds", default="")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--selection-lock", type=Path, default=None)
    args = parser.parse_args()
    root = args.output_root.resolve()
    matrix_root = root / "full_highres_matrix"
    matrix_root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "full_highres_matrix_route_plan.csv"
    if not plan_path.is_file():
        plan_path = matrix_root / "full_highres_matrix_route_plan.csv"
    failures: list[str] = []
    if not plan_path.is_file():
        failures.append(f"missing route plan: {plan_path}")
        plan = pd.DataFrame()
    else:
        plan = pd.read_csv(plan_path)
    expected_routes = set(plan["route"].astype(str)) if "route" in plan.columns else set()
    if not plan.empty and len(expected_routes) != len(plan):
        failures.append("route plan contains duplicate route names")
    if args.selection_lock is not None:
        if not args.selection_lock.is_file():
            failures.append(f"missing selection lock: {args.selection_lock}")
        else:
            try:
                selection_lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
                selected_routes = set(selection_lock.get("selected_routes", []))
                if selection_lock.get("status") != "locked" or selection_lock.get("test_metrics_used") is not False:
                    failures.append("selection lock is not a validation-only lock")
                if not selected_routes:
                    failures.append("selection lock contains no selected routes")
                expected_routes = selected_routes
            except Exception as exc:
                failures.append(f"cannot read selection lock: {exc}")
    locked, split_audit = locked_split_audit()
    if split_audit["n_images"] <= 0 or split_audit["n_patient_groups"] <= 0:
        failures.append(f"locked split is empty: {split_audit['n_images']} images/{split_audit['n_patient_groups']} groups")
    if (
        split_audit["patient_groups_crossing_splits"]
        or split_audit["exact_duplicate_links_crossing_splits"]
        or split_audit["near_duplicate_links_crossing_splits"]
    ):
        failures.append("locked split has a cross-split patient, duplicate, or near-duplicate group")
    expected_seeds = {int(value) for value in args.expected_seeds.split(",") if value.strip()} if args.expected_seeds else set()
    route_dirs = sorted(
        path for path in matrix_root.glob("seed_*/*")
        if path.is_dir()
        and (path / "run_manifest.json").is_file()
        and (not expected_routes or path.name in expected_routes)
    )
    observed_seeds = {int(path.parent.name.replace("seed_", "")) for path in route_dirs if path.parent.name.startswith("seed_")}
    if expected_seeds and observed_seeds != expected_seeds:
        failures.append(f"observed seeds {sorted(observed_seeds)} differ from expected {sorted(expected_seeds)}")
    if not args.bootstrap:
        failures.append("formal matrix validation requires patient-group bootstrap; pass --bootstrap")
    for seed in sorted(observed_seeds):
        observed_routes_for_seed = {
            path.name
            for path in route_dirs
            if path.parent.name == f"seed_{seed}"
        }
        if expected_routes and observed_routes_for_seed != expected_routes:
            missing_routes = sorted(expected_routes - observed_routes_for_seed)
            extra_routes = sorted(observed_routes_for_seed - expected_routes)
            failures.append(
                f"seed {seed}: observed route set differs from plan; "
                f"missing={missing_routes[:5]} extra={extra_routes[:5]}"
            )
        generation_methods = set(plan.loc[plan.get("ratio", pd.Series(dtype=float)).astype(float).gt(0), "generation_method"].astype(str)) if not plan.empty else set()
        for method in generation_methods:
            pool_root = root / ("synthetic" if method == "gan" else "synthetic_diffusion") / f"seed_{seed}"
            for name in ["generation_quality_metrics.csv", "generation_quality_complete.json"]:
                if not (pool_root / name).is_file() or (pool_root / name).stat().st_size == 0:
                    failures.append(f"seed {seed} {method}: missing {name}")
    recomputed, reported, bootstrap_frames = [], [], []
    for route_dir in route_dirs:
        current, old, route_failures = recompute_route(route_dir, locked)
        recomputed.extend(current)
        reported.extend(old)
        failures.extend(route_failures)
        if args.bootstrap and not route_failures:
            bootstrap_frames.append(patient_bootstrap(route_dir, args.n_bootstrap))
    representation_fingerprints: dict[tuple[str, str], set[str]] = {}
    representation_checkpoint_hashes: dict[tuple[str, str], set[str]] = {}
    for route_dir in route_dirs:
        try:
            manifest = json.loads((route_dir / "run_manifest.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        key = manifest.get("representation_key")
        fingerprint = manifest.get("representation_fingerprint")
        if key and fingerprint:
            representation_fingerprints.setdefault((route_dir.parent.name, str(key)), set()).add(str(fingerprint))
        checkpoint_hash = manifest.get("representation_checkpoint_sha256")
        if key and checkpoint_hash:
            representation_checkpoint_hashes.setdefault((route_dir.parent.name, str(key)), set()).add(str(checkpoint_hash))
    for (seed_name, key), fingerprints in representation_fingerprints.items():
        if len(fingerprints) != 1:
            failures.append(
                f"{seed_name}: shared representation key has inconsistent fingerprints: {key}"
            )
    for (seed_name, key), hashes in representation_checkpoint_hashes.items():
        if len(hashes) != 1:
            failures.append(
                f"{seed_name}: shared representation key has inconsistent checkpoint hashes: {key}"
            )
    recomputed_frame = pd.DataFrame(recomputed)
    reported_frame = pd.DataFrame(reported)
    recomputed_frame.to_csv(matrix_root / "full_matrix_metric_recomputation.csv", index=False)
    reported_frame.to_csv(matrix_root / "full_matrix_reported_metric_rows.csv", index=False)
    if bootstrap_frames:
        bootstrap_frame = pd.concat(bootstrap_frames, ignore_index=True)
        bootstrap_frame.to_csv(matrix_root / "full_matrix_patient_group_bootstrap.csv", index=False)
        if args.selection_lock is not None:
            bootstrap_frame.to_csv(root / "patient_group_bootstrap.csv", index=False)
    identity_rows = []
    for route_dir in route_dirs:
        prediction_path = route_dir / "predictions.csv"
        if not prediction_path.is_file():
            continue
        predictions = pd.read_csv(prediction_path)
        route = route_dir.name
        route_metrics = recomputed_frame[recomputed_frame["route"].astype(str).eq(route)] if not recomputed_frame.empty else pd.DataFrame()
        for point in sorted(route_metrics.get("operating_point", pd.Series(dtype=str)).dropna().unique()):
            val_row = route_metrics[(route_metrics["split"] == "val") & (route_metrics["operating_point"] == point)]
            test_row = route_metrics[(route_metrics["split"] == "test1") & (route_metrics["operating_point"] == point)]
            if len(val_row) != 1 or len(test_row) != 1:
                continue
            val_predictions = predictions[predictions["split"].astype(str).eq("val")]["prob_malignant"].to_numpy()
            test_predictions = predictions[predictions["split"].astype(str).eq("test1")]["prob_malignant"].to_numpy()
            image_ids_identical = set(predictions[predictions["split"].astype(str).eq("val")]["image_id"]) == set(predictions[predictions["split"].astype(str).eq("test1")]["image_id"])
            prediction_vectors_identical = len(val_predictions) == len(test_predictions) and np.array_equal(val_predictions, test_predictions)
            for metric in METRICS:
                val_value = val_row.iloc[0][metric]
                test_value = test_row.iloc[0][metric]
                identity_rows.append({
                    "route": route,
                    "operating_point": point,
                    "metric": metric,
                    "val_value": val_value,
                    "test1_value": test_value,
                    "exactly_equal": compare_float(val_value, test_value),
                    "equal_after_3_decimals": round(float(val_value), 3) == round(float(test_value), 3) if pd.notna(val_value) and pd.notna(test_value) else False,
                    "val_confusion": ";".join(f"{key}={val_row.iloc[0][key]}" for key in ["tp", "fp", "tn", "fn"]),
                    "test1_confusion": ";".join(f"{key}={test_row.iloc[0][key]}" for key in ["tp", "fp", "tn", "fn"]),
                    "prediction_vectors_identical": prediction_vectors_identical,
                    "image_ids_identical": image_ids_identical,
                    "explanation_required": compare_float(val_value, test_value) or (round(float(val_value), 3) == round(float(test_value), 3) if pd.notna(val_value) and pd.notna(test_value) else False),
                })
    pd.DataFrame(identity_rows).to_csv(matrix_root / "metric_identity_audit.csv", index=False)
    summary = recomputed_frame[
        recomputed_frame.get("operating_point", pd.Series(dtype=str)).eq("sens90")
        & recomputed_frame.get("split", pd.Series(dtype=str)).eq("test1")
    ].copy() if not recomputed_frame.empty else pd.DataFrame()
    summary.to_csv(matrix_root / "full_matrix_test1_sens90_summary.csv", index=False)
    expected_per_seed = len(expected_routes) if args.selection_lock is not None else len(plan)
    complete_routes = len(route_dirs)
    if expected_seeds:
        expected_total = expected_per_seed * len(expected_seeds)
        if complete_routes != expected_total:
            failures.append(f"complete routes={complete_routes} expected={expected_total}")
    elif plan.empty:
        failures.append("route plan is empty")
    else:
        if complete_routes != expected_per_seed * max(len(observed_seeds), 1):
            failures.append(f"complete routes={complete_routes} does not match plan x observed seeds={expected_per_seed * max(len(observed_seeds), 1)}")
    payload = {
        "status": "complete" if not failures else "failed",
        "expected_routes_per_seed": expected_per_seed,
        "complete_routes": complete_routes,
        "expected_seeds": sorted(expected_seeds),
        "observed_seeds": sorted(observed_seeds),
        "expected_route_names": sorted(expected_routes),
        "split_audit": split_audit,
        "metric_rows_recomputed": int(len(recomputed_frame)),
        "shared_representation_groups": int(len(representation_fingerprints)),
        "failures": failures,
    }
    (matrix_root / "full_matrix_validation.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
