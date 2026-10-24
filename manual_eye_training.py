"""Load only the new manually cropped four-way cohort. Author: Jia Rong."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import pandas as pd


SPLIT_FILES = {"train": "retrospective_train.csv", "validation": "retrospective_validation.csv",
               "test": "retrospective_test.csv", "prospective_test": "prospective_test.csv"}


def audit_cross_split_visual_candidates(frame: pd.DataFrame, root: Path) -> tuple[int, list[dict]]:
    """Screen dHash candidates with SSIM; low-distance hashes alone are not duplicates."""
    from PIL import Image
    import numpy as np
    from skimage.metrics import structural_similarity

    records = frame.reset_index(drop=True)
    hashes = [int(value, 16) for value in records["crop_dhash"].astype(str)]
    splits = records["split"].astype(str).tolist()
    image_ids = records["image_id"].astype(str).tolist()
    crop_paths = records["crop_path"].astype(str).tolist()
    candidates = []
    confirmed = 0
    for left, right in itertools.combinations(range(len(records)), 2):
        if splits[left] == splits[right]:
            continue
        distance = (hashes[left] ^ hashes[right]).bit_count()
        if distance <= 2:
            candidates.append((left, right, distance))
    visual_candidates = []
    for left, right, distance in candidates:
        images = []
        for index in (left, right):
            with Image.open(root / crop_paths[index]) as image:
                images.append(np.asarray(image.convert("RGB").resize((224, 224))))
        score = float(structural_similarity(images[0], images[1], channel_axis=2, data_range=255))
        is_duplicate = score >= 0.90
        confirmed += int(is_duplicate)
        visual_candidates.append({
            "image_id_a": image_ids[left],
            "split_a": splits[left],
            "image_id_b": image_ids[right],
            "split_b": splits[right],
            "dhash_hamming": int(distance),
            "ssim_224": score,
            "confirmed_near_duplicate": is_duplicate,
        })
    return confirmed, visual_candidates


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manual_data(root: Path, base):
    root = root.resolve()
    definition = json.loads((root / "splits/split_definition.json").read_text(encoding="utf-8"))
    if not definition.get("patient_level_independence_fully_verified"):
        raise ValueError("Resolve or exclude unidentified patient groups before formal training")
    frames = []
    for split, name in SPLIT_FILES.items():
        path = root / "splits" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != definition["files"][name]["sha256"]:
            raise ValueError(f"Split index changed: {name}")
        frame = pd.read_csv(path, dtype={"patient_group": str, "image_id": str,
                                         "crop_dhash": str, "crop_sha256": str, "original_md5": str})
        if frame.empty or not frame["split"].eq(split).all():
            raise ValueError(f"Invalid split membership: {name}")
        if len(frame) != definition["files"][name]["crops"]:
            raise ValueError(f"Unexpected split size: {name}")
        if frame["patient_identity_basis"].eq("unkeyed_singleton").any():
            raise ValueError("Unidentified old material entered the formal cohort")
        frame["absolute_path"] = [str((root / value).resolve()) for value in frame["crop_path"]]
        for _, row in frame.iterrows():
            value = row["absolute_path"]
            if not Path(value).is_relative_to(root) or not Path(value).is_file():
                raise ValueError(f"Missing crop or path outside the dataset: {value}")
            if not os.environ.get("MANUAL_EYE_DATA_VERIFIED"):
                from PIL import Image
                if file_sha256(Path(value)) != str(row["crop_sha256"]).lower():
                    raise ValueError(f"Crop SHA-256 differs from the split index: {value}")
                with Image.open(value) as image:
                    image.verify()
        if (split == "prospective_test" and not frame["source_batch"].eq(7).all()) or (
                split != "prospective_test" and not frame["source_batch"].between(1, 6).all()):
            raise ValueError("Batch-seven boundary was violated")
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    expected_names = frame["label"].map({0: "benign", 1: "malignant"})
    if (frame["image_id"].duplicated().any() or not frame["label"].isin([0, 1]).all()
            or not expected_names.equals(frame["class_name"].astype(str))):
        raise ValueError("Invalid image identity or clinical label")
    for field in ["patient_group", "original_md5", "crop_sha256"]:
        if frame[field].isna().any() or frame.groupby(field)["split"].nunique().gt(1).any():
            raise ValueError(f"Missing identity or split overlap: {field}")
    if frame.groupby("patient_group")["label"].nunique().gt(1).any():
        raise ValueError("Conflicting patient labels")
    frame["split"] = frame["split"].replace({"validation": "val", "test": "test1", "prospective_test": "source_holdout"})
    frame["source_holdout"] = frame["split"].eq("source_holdout")
    frame["original_split"] = frame["source_holdout"].map({True: "test", False: "raw"})
    frame["synthetic"] = False
    frame["label_name"] = frame["label"].map({0: "benign", 1: "malignant"})
    frame["filename"] = frame["crop_path"].map(lambda value: Path(value).name)
    frame["md5"] = frame["original_md5"]
    frame["dhash"] = frame["crop_dhash"]
    confirmed_near_duplicates, visual_candidates = audit_cross_split_visual_candidates(frame, root)
    definition_text = "supplied clinical case directories plus retained manual patient links"
    frame.attrs["patient_group_definition"] = definition_text
    locked_train = frame[frame["split"].eq("train")].copy()
    counts = locked_train.groupby("patient_group", sort=True).agg(
        n_images=("image_id", "size"), benign=("label", lambda x: int(x.eq(0).sum())),
        malignant=("label", lambda x: int(x.eq(1).sum())), mixed_labels=("label", lambda x: x.nunique() > 1)).reset_index()
    assignment = base.assign_groups(counts, {"train_fit": 0.85, "train_earlystop": 0.15}, seed=1729)
    locked_train["training_subsplit"] = locked_train["patient_group"].map(assignment)
    splits = {"locked_train": locked_train,
              "train": locked_train[locked_train["training_subsplit"].eq("train_fit")].copy(),
              "early_stop": locked_train[locked_train["training_subsplit"].eq("train_earlystop")].copy()}
    splits.update({name: frame[frame["split"].eq(name)].copy() for name in ["val", "test1", "source_holdout"]})
    if any(part.empty or set(part["label"]) != {0, 1} for part in splits.values()):
        raise ValueError("Each training/evaluation partition must contain both clinical classes")
    audit = {"group_field": "patient_group", "n_images": len(frame),
             "n_patient_groups": frame["patient_group"].nunique(), "patient_group_definition": definition_text,
             "patient_groups_crossing_splits": 0, "exact_duplicate_links_crossing_splits": 0,
             "near_duplicate_links_crossing_splits": confirmed_near_duplicates,
             "cross_split_dhash_candidates": len(visual_candidates),
             "cross_split_dhash_candidates_ssim_audited": True,
             "cross_split_dhash_candidate_max_ssim": max((item["ssim_224"] for item in visual_candidates), default=None),
             "cross_split_dhash_candidate_details": visual_candidates,
             "mixed_label_patient_groups": 0, "source_holdout_images": len(splits["source_holdout"]),
             "source_holdout_patient_groups": splits["source_holdout"]["patient_group"].nunique(),
             "cohort_names": {"test1": "Retrospective Cohort Test", "source_holdout": "Prospective Cohort Test"},
             "later_cohort_design": "same_hospital_temporal_validation",
             "split_counts": {"|".join(map(str, key)): int(value) for key, value in frame.groupby(["split", "label_name"]).size().items()}}
    return frame, audit, splits
