#!/usr/bin/env python3
"""Manifest-driven high-resolution SY-CNN+ experiment matrix.

This entry point implements the fixed-cohort, patient-grouped experiment
protocol. Diffusion is implemented locally so the package does not depend on
an optional external diffusion library. All downstream
heads use a shared trainable 512-D projection of the native backbone
representation.

The matrix is intentionally opt-in. Command-line selectors are accepted only
to validate the locked protocol; formal runs must execute the complete selected
paper matrix. A dry run should be used to inspect the route count before a GPU
submission. All formal routes use the locked split and write independent,
resumable artifacts under ``OUTPUT_ROOT/full_highres_matrix``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


BUNDLE = Path(__file__).resolve().parent
os.environ.setdefault("RUN_STAGE", "full_matrix")
# Formal matrix runs stay locked to 256 px. The generator-quality pilot may
# explicitly opt into a supported experimental size in its run config.
ALLOW_EXPERIMENTAL_GENERATOR_SIZE = os.environ.get(
    "ALLOW_EXPERIMENTAL_GENERATOR_SIZE", "0"
).strip().lower() in {"1", "true", "yes"}
os.environ["IMAGE_SIZE"] = "224"
if ALLOW_EXPERIMENTAL_GENERATOR_SIZE:
    selected_generator_size = int(os.environ.get("GAN_SIZE", "256"))
    if selected_generator_size not in {128, 256, 512}:
        raise ValueError(f"Experimental generator size must be 128, 256, or 512, got {selected_generator_size}")
    os.environ["GAN_SIZE"] = str(selected_generator_size)
else:
    os.environ["GAN_SIZE"] = "256"
os.environ["BACKBONE_INIT"] = "imagenet"
os.environ.setdefault("GAN_POOL_PER_CLASS", "8192")
os.environ.setdefault("DIFFUSION_POOL_PER_CLASS", "8192")
sys.path.insert(0, str(BUNDLE))

import reproduce_gan_resnet50_xgb_patient224 as base  # noqa: E402


IMAGE_SIZE = 224
GENERATOR_SIZE = int(os.environ.get("GAN_SIZE", "256"))
PROTOCOL_CONTRACT_PATH = BUNDLE / (
    "manual_eye_training_contract.json"
    if os.environ.get("MANUAL_EYE_DATASET_ROOT")
    else "full_highres_matrix_contract.json"
)
PROTOCOL_CONTRACT = json.loads(PROTOCOL_CONTRACT_PATH.read_text(encoding="utf-8"))
DEFAULT_METHODS = tuple(
    item["name"]
    for item in PROTOCOL_CONTRACT["generation_methods"]
    if item["name"] != "real_only"
)
DEFAULT_BACKBONES = tuple(PROTOCOL_CONTRACT["backbones"])
DEFAULT_HEADS = tuple(PROTOCOL_CONTRACT["downstream_heads"])
DEFAULT_RATIOS = tuple(float(value) for value in PROTOCOL_CONTRACT["synthetic_ratios"])
DEFAULT_SEED = int(os.environ.get("RUN_SEED", "42"))
PROJECTION_DIM = 512
PATIENT_GROUP_FIELD = "patient_group"
SUPPORTED_PAPER_BACKBONES = set(DEFAULT_BACKBONES)
NATIVE_FEATURE_DIMS = {"resnet50": 2048}
if GENERATOR_SIZE != base.GAN_SIZE:
    raise RuntimeError("Base GAN size and generator-module image size diverged")


def parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not value.strip():
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ratios(value: str | None) -> list[float]:
    if value is None or not value.strip():
        return list(DEFAULT_RATIOS)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_quality_tokens(value: str | None, default: tuple[str, ...]) -> list[bool]:
    """Parse LidSentry flags without silently coercing unknown values to off."""

    tokens = parse_csv(value, default)
    mapping = {
        "off": False,
        "false": False,
        "0": False,
        "on": True,
        "true": True,
        "1": True,
    }
    unknown = sorted({token.lower() for token in tokens} - set(mapping))
    if unknown:
        raise ValueError(f"Unknown quality values {unknown}; use off/on")
    return [mapping[token.lower()] for token in tokens]


def stable_token(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def ratio_token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def route_name(spec: dict) -> str:
    method = spec["generation_method"]
    ratio = ratio_token(float(spec["ratio"]))
    strategy = spec["strategy"]
    quality = "lidsentry" if spec["lidsentry"] else "nolidsentry"
    strong = "_strongconv" if spec.get("strong_conventional") else ""
    return (
        f"{method}_{spec['backbone']}_{spec['head']}_n{ratio}_"
        f"{strategy}_{quality}{strong}_size{IMAGE_SIZE}"
    )


def representation_key(spec: dict) -> str:
    """Identify one shared backbone/projection training unit across heads."""

    values = {
        key: spec[key]
        for key in [
            "paper", "generation_method", "backbone", "ratio", "strategy",
            "lidsentry", "strong_conventional",
        ]
        if key in spec
    }
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def representation_artifact_dir(spec: dict, seed: int) -> Path:
    """Return the disk-level representation directory shared by all heads."""

    return (
        base.OUT
        / "full_highres_matrix"
        / f"seed_{seed}"
        / "_representations"
        / stable_token(representation_key(spec))
    )


def representation_artifact_relpath(spec: dict) -> str:
    """Store a portable path relative to a seed directory in each manifest."""

    return str(Path("_representations") / stable_token(representation_key(spec)))


def cohort_defaults(cohort: str) -> dict[str, object]:
    normalized = cohort.strip().lower()
    if normalized != "retrospective_cohort":
        raise ValueError("This entry point supports the retrospective-cohort protocol only.")
    try:
        locked = PROTOCOL_CONTRACT["route_selector_defaults"][normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported fixed-split protocol: {cohort}.") from exc
    return {
        "methods": tuple(locked["methods"]),
        "backbones": tuple(locked["backbones"]),
        "heads": tuple(locked["heads"]),
        "ratios": tuple(float(value) for value in locked["ratios"]),
        "quality": tuple("on" if value else "off" for value in locked["lidsentry"]),
        "strategies": tuple(locked["strategies"]),
        "include_strong": bool(locked["strong_conventional_routes"]),
    }


def validate_cohort_protocol(
    cohort: str,
    methods: list[str],
    backbones: list[str],
    heads: list[str],
    ratios: list[float],
    quality_values: list[bool],
    strategies: list[str],
    include_strong: bool,
) -> None:
    defaults = cohort_defaults(cohort)
    unsupported = sorted(set(backbones) - SUPPORTED_PAPER_BACKBONES)
    if unsupported:
        raise ValueError(
            "The formal high-resolution protocols allow only ResNet50; "
            f"unsupported backbones: {unsupported}"
        )
    expected_backbones = set(defaults["backbones"])
    expected_methods = set(defaults["methods"])
    expected_heads = set(defaults["heads"])
    expected_ratios = {float(value) for value in defaults["ratios"]}
    expected_quality = {value == "on" for value in defaults["quality"]}
    expected_strategies = set(defaults["strategies"])
    if set(methods) != expected_methods:
        raise ValueError(f"{cohort} requires generation methods {sorted(expected_methods)}")
    if set(backbones) != expected_backbones:
        raise ValueError(f"{cohort} requires backbones {sorted(expected_backbones)}")
    if set(heads) != expected_heads:
        raise ValueError(f"{cohort} requires heads {sorted(expected_heads)}")
    if {float(value) for value in ratios} != expected_ratios:
        raise ValueError(f"{cohort} requires ratios {sorted(expected_ratios)}")
    if set(quality_values) != expected_quality:
        raise ValueError(f"{cohort} requires LidSentry values {sorted(expected_quality)}")
    if set(strategies) != expected_strategies:
        raise ValueError(f"{cohort} requires strategies {sorted(expected_strategies)}")
    if not include_strong:
        raise ValueError(f"{cohort} requires the declared conventional strong controls")


def build_specs(
    methods: list[str],
    backbones: list[str],
    heads: list[str],
    ratios: list[float],
    quality_values: list[bool],
    strategies: list[str],
    include_strong: bool,
) -> list[dict]:
    specs: list[dict] = []
    for ratio in ratios:
        generation_methods = ["real_only"] if ratio == 0 else methods
        for method in generation_methods:
            for backbone in backbones:
                for head in heads:
                    for lidsentry in quality_values:
                        for strategy in strategies:
                            if strategy == "balanced" and ratio != 4:
                                continue
                            specs.append({
                                "generation_method": method,
                                "backbone": backbone,
                                "head": head,
                                "ratio": float(ratio),
                                "strategy": strategy,
                                "lidsentry": bool(lidsentry),
                                "strong_conventional": False,
                            })
    if include_strong:
        # Strong controls are matched by generator family and head.  The
        # diffusion controls are important: without them, a positive GAN
        # result cannot be separated from a generic synthetic-data effect.
        strong_specs = [
            {"generation_method": "real_only", "backbone": "resnet50", "head": "cnn", "ratio": 0.0},
            {"generation_method": "real_only", "backbone": "resnet50", "head": "xgboost", "ratio": 0.0},
            {"generation_method": "gan", "backbone": "resnet50", "head": "cnn", "ratio": 4.0},
            {"generation_method": "gan", "backbone": "resnet50", "head": "xgboost", "ratio": 4.0},
            {"generation_method": "diffusion", "backbone": "resnet50", "head": "cnn", "ratio": 4.0},
            {"generation_method": "diffusion", "backbone": "resnet50", "head": "xgboost", "ratio": 4.0},
        ]
        for item in strong_specs:
            for strategy in ["original_ratio"]:
                for lidsentry in [False]:
                    specs.append({
                        **item,
                        "strategy": strategy,
                        "lidsentry": lidsentry,
                        "strong_conventional": True,
                    })
    unique: dict[str, dict] = {}
    for spec in specs:
        unique[route_name(spec)] = spec
    return [unique[key] for key in sorted(unique)]


def load_locked_data() -> tuple[pd.DataFrame, dict, dict[str, pd.DataFrame]]:
    if os.environ.get("MANUAL_EYE_DATASET_ROOT"):
        from manual_eye_training import load_manual_data
        return load_manual_data(Path(os.environ["MANUAL_EYE_DATASET_ROOT"]), base)
    canonical_root = os.environ.get("CANONICAL_IMAGE_ROOT")
    manifest_value = os.environ.get("MANIFEST_PATH")
    split_value = os.environ.get("LOCKED_SPLIT")
    if not canonical_root or not manifest_value or not split_value:
        raise RuntimeError(
            "Formal runs require CANONICAL_IMAGE_ROOT, MANIFEST_PATH, and LOCKED_SPLIT "
            "for the frozen raw-tumor cohort; legacy image archives are disabled."
        )
    manifest_path = Path(manifest_value).resolve()
    split_path = Path(split_value).resolve()
    if not manifest_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(
            "The formal input assets are missing: "
            f"manifest={manifest_path}, split={split_path}"
        )
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
                "The formal matrix received an input manifest or split outside "
                "the frozen current cohort: "
                + json.dumps(mismatches, sort_keys=True)
            )
    data_root = Path(canonical_root)
    manifest = base.load_precomputed_manifest(data_root)
    if manifest is None:
        raise RuntimeError("manifest_raw.csv is required for the full high-resolution matrix")
    frame, audit = base.load_locked_split(manifest)
    frame = frame.copy()
    expected = PROTOCOL_CONTRACT["data_split"]
    observed = {
        "images": int(len(frame)),
        "patient_groups": int(frame[PATIENT_GROUP_FIELD].nunique()),
    }
    mismatches = {
        key: {"expected": int(expected[key]), "observed": value}
        for key, value in observed.items()
        if int(expected[key]) != value
    }
    if mismatches:
        raise RuntimeError(
            "The formal matrix received a stale grouping or legacy manifest: "
            + json.dumps(mismatches, sort_keys=True)
        )
    expected_split_counts = {
        "train": int(expected["train_images"]),
        "val": int(expected["validation_images"]),
        "test1": int(expected["internal_test_images"]),
    }
    observed_split_counts = (
        frame["split"].astype(str).str.lower().replace({"test": "test1"})
        .value_counts().to_dict()
    )
    if any(
        observed_split_counts.get(key, 0) != value
        for key, value in expected_split_counts.items()
    ):
        raise RuntimeError(
            "The formal matrix received a stale locked split: "
            + json.dumps(
                {"expected": expected_split_counts, "observed": observed_split_counts},
                sort_keys=True,
            )
        )
    if frame["image_id"].duplicated().any():
        raise RuntimeError("The formal matrix input contains duplicate image_id values")
    frame.attrs["patient_group_definition"] = audit.get("patient_group_definition", "authoritative manifest patient_group")
    frame["synthetic"] = False
    # A legacy source holdout is used only when the manifest explicitly carries
    # an original ``test`` source split.  The raw tumor index is a single raw
    # cohort and therefore has no source holdout.
    source_flags = frame.assign(_is_original_test=frame["original_split"].astype(str).str.lower().eq("test"))
    source_groups = set(source_flags.loc[source_flags["_is_original_test"], "patient_group"])
    frame["source_holdout"] = frame["patient_group"].isin(source_groups)
    development = frame.loc[~frame["source_holdout"]].copy()
    # Keep the locked validation/test sets untouched, but split the development
    # training groups once more so early stopping cannot consume the route-
    # selection validation fold.  The split is deterministic and group-safe.
    locked_train = development.loc[development["split"].eq("train")].copy()
    group_counts = (
        locked_train.groupby(PATIENT_GROUP_FIELD, sort=True)
        .agg(
            n_images=("image_id", "size"),
            benign=("label", lambda values: int((values.astype(int) == 0).sum())),
            malignant=("label", lambda values: int((values.astype(int) == 1).sum())),
            mixed_labels=("label", lambda values: bool(values.astype(int).nunique() > 1)),
        )
        .reset_index()
    )
    training_assignment = base.assign_groups(
        group_counts,
        {"train_fit": 0.85, "train_earlystop": 0.15},
        seed=1729,
    )
    locked_train["training_subsplit"] = locked_train[PATIENT_GROUP_FIELD].map(training_assignment)
    if locked_train["training_subsplit"].isna().any():
        raise RuntimeError("Some locked training patient groups were not assigned to an early-stop split")
    train_fit = locked_train.loc[locked_train["training_subsplit"].eq("train_fit")].copy()
    train_earlystop = locked_train.loc[locked_train["training_subsplit"].eq("train_earlystop")].copy()
    if train_fit.empty or train_earlystop.empty:
        raise RuntimeError("The deterministic training/early-stop split is empty")
    splits = {
        "train": train_fit,
        "early_stop": train_earlystop,
        "locked_train": locked_train,
        "val": development.loc[development["split"].eq("val")].copy(),
        "test1": development.loc[development["split"].eq("test1")].copy(),
        "source_holdout": frame.loc[frame["source_holdout"]].copy(),
    }
    return frame, audit, splits


def make_projection_head(nn, native_dim: int, projection_dim: int = PROJECTION_DIM):
    """Project every backbone to one shared trainable lesion representation."""

    class ProjectionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Sequential(
                nn.Linear(native_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Dropout(0.30),
            )
            self.classifier = nn.Linear(projection_dim, 1)

        def forward(self, values):
            return self.classifier(self.embedding(values))

    return ProjectionHead()


class VanillaCNN:
    """Legacy architecture helper retained for explicit non-paper experiments."""

    def __init__(self, torch_module):
        nn = torch_module.nn
        self._nn = nn

    def build(self):
        nn = self._nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                )
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.feature_projection = make_projection_head(nn, 256)
                self._native_feature_dim = 256
                self._projection_dim = PROJECTION_DIM
                self._pretrained_weights_used = False
                self._initialization = "random_architecture"
                self._full_backbone_name = "vanilla_cnn"

            def native_features(self, values):
                return self.pool(self.features(values)).flatten(1)

            def forward_features(self, values):
                return self.feature_projection.embedding(self.native_features(values))

            def forward(self, values):
                return self.feature_projection(self.native_features(values))

        return Model()


def build_model(name: str):
    import torch.nn as nn
    from torchvision import models

    normalized = name.lower().replace("-", "_")
    if normalized == "vanilla_cnn":
        return VanillaCNN(__import__("torch")).build()
    if normalized == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        native_dim = model.fc.in_features
        model.fc = make_projection_head(nn, native_dim)
    elif normalized == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        native_dim = model.fc.in_features
        model.fc = make_projection_head(nn, native_dim)
    elif normalized == "vgg11":
        model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
        native_dim = model.classifier[0].in_features
        model.classifier = make_projection_head(nn, native_dim)
    elif normalized == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        native_dim = model.classifier[0].in_features
        model.classifier = make_projection_head(nn, native_dim)
    elif normalized in {"efficientnet_b0", "efficientnetb0"}:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        native_dim = model.classifier[-1].in_features
        model.classifier = make_projection_head(nn, native_dim)
    else:
        raise ValueError(f"Unsupported full-matrix backbone: {name}")
    model._native_feature_dim = int(native_dim)
    model._projection_dim = PROJECTION_DIM
    model._pretrained_weights_used = True
    model._initialization = "imagenet"
    model._full_backbone_name = normalized
    return model


def model_features(model, values):
    name = getattr(model, "_full_backbone_name", "")
    if hasattr(model, "forward_features"):
        return model.forward_features(values)
    if name.startswith("resnet"):
        values = model.conv1(values)
        values = model.bn1(values)
        values = model.relu(values)
        values = model.maxpool(values)
        values = model.layer1(values)
        values = model.layer2(values)
        values = model.layer3(values)
        values = model.layer4(values)
        native = model.avgpool(values).flatten(1)
        return model.fc.embedding(native)
    if name.startswith("vgg"):
        values = model.features(values)
        values = model.avgpool(values)
        native = values.flatten(1)
        return model.classifier.embedding(native)
    if name.startswith("efficientnet"):
        values = model.features(values)
        values = model.avgpool(values)
        native = values.flatten(1)
        return model.classifier.embedding(native)
    raise ValueError(f"Cannot extract features for {name}")


def predict_model(model, frame: pd.DataFrame, quality: bool, device):
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        base.ImageDataset(frame, IMAGE_SIZE, "none", False, quality).dataset,
        batch_size=base.BATCH_SIZE,
        shuffle=False,
        num_workers=base.NUM_WORKERS,
    )
    model.eval()
    probabilities, labels, ids = [], [], []
    with torch.no_grad():
        for values, target, image_ids, _ in loader:
            logits = model(values.to(device)).view(-1)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(target.numpy().astype(int).tolist())
            ids.extend(list(image_ids))
    return np.asarray(probabilities), np.asarray(labels), ids


def extract_features(model, frame: pd.DataFrame, quality: bool, device):
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        base.ImageDataset(frame, IMAGE_SIZE, "none", False, quality).dataset,
        batch_size=base.BATCH_SIZE,
        shuffle=False,
        num_workers=base.NUM_WORKERS,
    )
    model.eval()
    features, labels, ids, quality_values = [], [], [], []
    with torch.no_grad():
        for values, target, image_ids, q in loader:
            features.append(model_features(model, values.to(device)).cpu().numpy())
            labels.extend(target.numpy().astype(int).tolist())
            ids.extend(list(image_ids))
            quality_values.extend(q.numpy().astype(float).tolist())
    result = np.concatenate(features, axis=0).astype(np.float32) if features else np.empty((0, 1), dtype=np.float32)
    if quality:
        result = np.column_stack([result, np.asarray(quality_values, dtype=np.float32)])
    return result, np.asarray(labels), ids


def train_model(
    train_frame: pd.DataFrame,
    early_stop_frame: pd.DataFrame,
    backbone: str,
    quality: bool,
    strong_conventional: bool,
    seed: int,
    checkpoint_dir: Path,
    configuration_fingerprint: str,
    real_training_size: int | None = None,
):
    import torch
    from torch.utils.data import DataLoader

    base.seed_everything(seed)
    device = torch.device(base.device_name())
    model = build_model(backbone).to(device)
    if backbone != "vanilla_cnn" and not bool(getattr(model, "_pretrained_weights_used", False)):
        raise RuntimeError(f"Expected ImageNet initialization for {backbone}")
    train_dataset = base.ImageDataset(
        train_frame,
        IMAGE_SIZE,
        "strong" if strong_conventional else ("degradation" if quality else "legacy"),
        True,
        quality,
        degradation_seed=seed,
    )
    train_loader = DataLoader(train_dataset.dataset, batch_size=base.BATCH_SIZE, shuffle=True, num_workers=base.NUM_WORKERS)
    updates_per_epoch = max(1, math.ceil(int(real_training_size or len(train_frame)) / base.BATCH_SIZE))
    early_stop_loader = DataLoader(
        base.ImageDataset(early_stop_frame, IMAGE_SIZE, "none", False, quality).dataset,
        batch_size=base.BATCH_SIZE,
        shuffle=False,
        num_workers=base.NUM_WORKERS,
    )
    positive = float((train_frame["label"] == 1).sum())
    negative = float((train_frame["label"] == 0).sum())
    class_weighted = bool(strong_conventional)
    pos_weight = torch.tensor([negative / max(positive, 1.0)], device=device) if class_weighted else None
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none" if quality else "mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "backbone_checkpoint.pt"
    best_state, best_score, best_epoch, stale, history = None, -np.inf, 0, 0, []
    start_epoch = 1
    if checkpoint_path.exists() and not base.FORCE_RERUN:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("configuration_fingerprint") == configuration_fingerprint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_state = checkpoint.get("best_state")
            best_score = float(checkpoint.get("best_score", -np.inf))
            best_epoch = int(checkpoint.get("best_epoch", 0))
            stale = int(checkpoint.get("stale", 0))
            history = list(checkpoint.get("history", []))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            if int(checkpoint.get("epoch", 0)) >= base.CNN_MIN_EPOCHS and stale >= base.PATIENCE:
                start_epoch = base.EPOCHS + 1
    for epoch in range(start_epoch, base.EPOCHS + 1):
        model.train()
        train_dataset.set_epoch(epoch)
        total_loss = 0.0
        processed_examples = 0
        train_iterator = iter(train_loader)
        for _ in range(updates_per_epoch):
            try:
                values, target, _, quality_values = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                values, target, _, quality_values = next(train_iterator)
            values, target, quality_values = values.to(device), target.to(device), quality_values.to(device)
            optimizer.zero_grad(set_to_none=True)
            if strong_conventional:
                mix_lambda = float(np.random.beta(0.4, 0.4))
                permutation = torch.randperm(values.size(0), device=device)
                mixed_values = mix_lambda * values + (1.0 - mix_lambda) * values[permutation]
                logits = model(mixed_values).view(-1)
                loss = mix_lambda * loss_fn(logits, target) + (1.0 - mix_lambda) * loss_fn(logits, target[permutation])
                quality_values = mix_lambda * quality_values + (1.0 - mix_lambda) * quality_values[permutation]
            else:
                logits = model(values).view(-1)
                loss = loss_fn(logits, target)
            if quality:
                loss = (loss * (0.75 + 0.5 * quality_values)).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(target)
            processed_examples += len(target)
        early_stop_prob, early_stop_y, _ = base.predict_backbone(model, early_stop_loader, device)
        score = base.safe_auc(early_stop_y, early_stop_prob)
        history.append({
            "epoch": epoch,
            "optimizer_updates": updates_per_epoch,
            "train_loss": total_loss / max(processed_examples, 1),
            "early_stop_auroc": score,
        })
        if np.isfinite(score) and score > best_score + base.CNN_MIN_DELTA:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        checkpoint = {
            "configuration_fingerprint": configuration_fingerprint,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_state": best_state,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "stale": stale,
            "history": history,
        }
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)
        if epoch >= base.CNN_MIN_EPOCHS and stale >= base.PATIENCE:
            break
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_epoch = int(history[-1]["epoch"]) if history else 0
        best_score = float(history[-1].get("early_stop_auroc", np.nan)) if history else np.nan
    model.load_state_dict(best_state)
    history_frame = pd.DataFrame(history)
    history_frame["selected_best_epoch"] = int(best_epoch)
    history_frame["selected_best_early_stop_auroc"] = float(best_score)
    # Keep the legacy column for downstream readers, but make the provenance
    # explicit so it cannot be mistaken for the formal validation fold.
    history_frame["selected_best_val_auroc"] = float(best_score)
    return model, history_frame, device


def load_representation_from_dir(spec: dict, representation_dir: Path):
    """Load the disk-level shared backbone/projection representation."""

    import torch

    checkpoint_path = representation_dir / "backbone_best_state.pt"
    history_path = representation_dir / "history.csv"
    if not checkpoint_path.is_file() or not history_path.is_file():
        raise RuntimeError(f"Cannot reuse missing representation artifacts from {representation_dir}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(spec["backbone"])
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(base.device_name())
    model = model.to(device)
    history = pd.read_csv(history_path)
    return model, history, device


def load_saved_representation(spec: dict, route_dir: Path):
    """Load a route's shared representation, with legacy route-local fallback."""

    manifest_path = route_dir / "run_manifest.json"
    representation_dir = route_dir
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative = manifest.get("representation_artifact_dir")
        if relative:
            representation_dir = route_dir.parent / Path(str(relative))
    return load_representation_from_dir(spec, representation_dir)


def save_shared_representation(
    spec: dict,
    seed: int,
    representation_dir: Path,
    model,
    history: pd.DataFrame,
    representation_fingerprint: str,
) -> None:
    """Persist one source-of-truth representation before fitting downstream heads."""

    import torch

    representation_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "backbone": spec["backbone"],
        "representation_key": representation_key(spec),
        "representation_fingerprint": representation_fingerprint,
    }, representation_dir / "backbone_best_state.pt")
    history.to_csv(representation_dir / "history.csv", index=False)
    (representation_dir / "representation_complete.json").write_text(json.dumps({
        "status": "complete",
        "seed": seed,
        "backbone": spec["backbone"],
        "representation_key": representation_key(spec),
        "representation_fingerprint": representation_fingerprint,
        "feature_dimension": PROJECTION_DIM,
    }, indent=2), encoding="utf-8")


class DiffusionDenoiser:
    def __init__(
        self,
        torch_module,
        base_channels: int = 32,
        timesteps: int = 100,
        structure_condition: bool = False,
    ):
        nn = torch_module.nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.timesteps = timesteps
                self.structure_condition = bool(structure_condition)
                self.time = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 256))
                self.label = nn.Embedding(2, 256)
                c0, c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
                self.in_conv = nn.Conv2d(3, c0, 3, padding=1)
                if self.structure_condition:
                    # The frozen teacher produces 3 soft maps (sclera,
                    # iris/pupil, eyelid) at 64 x 64.  Encoding them in a
                    # separate branch gives the denoiser an explicit spatial
                    # anatomy signal instead of asking a scalar loss to
                    # reconstruct the eye layout from noisy pixels.
                    self.structure_in = nn.Conv2d(3, c0, 3, padding=1)
                self.enc0 = nn.ModuleList([self.ResidualBlock(c0, c0), self.ResidualBlock(c0, c0)])
                self.down1 = nn.Conv2d(c0, c1, 4, 2, 1)
                self.enc1 = nn.ModuleList([self.ResidualBlock(c1, c1), self.ResidualBlock(c1, c1)])
                self.down2 = nn.Conv2d(c1, c2, 4, 2, 1)
                self.enc2 = nn.ModuleList([self.ResidualBlock(c2, c2), self.ResidualBlock(c2, c2)])
                self.down3 = nn.Conv2d(c2, c3, 4, 2, 1)
                self.enc3 = nn.ModuleList([self.ResidualBlock(c3, c3), self.ResidualBlock(c3, c3)])
                self.down4 = nn.Conv2d(c3, c3, 4, 2, 1)
                self.mid0 = self.ResidualBlock(c3, c3)
                self.mid_attention = self.AttentionBlock(c3, heads=4)
                self.mid1 = self.ResidualBlock(c3, c3)

                # Four skip-connected decoder stages recover spatial detail.
                # Nearest-neighbour upsampling followed by a learned convolution
                # avoids the smoothing of repeated bilinear interpolation.
                self.up3 = nn.Conv2d(c3, c3, 3, padding=1)
                self.dec3 = self.ResidualBlock(c3 * 2, c3)
                self.up2 = nn.Conv2d(c3, c2, 3, padding=1)
                self.dec2 = self.ResidualBlock(c2 * 2, c2)
                self.up1 = nn.Conv2d(c2, c1, 3, padding=1)
                self.dec1 = self.ResidualBlock(c1 * 2, c1)
                self.up0 = nn.Conv2d(c1, c0, 3, padding=1)
                self.dec0 = self.ResidualBlock(c0 * 2, c0)
                self.out_norm = nn.GroupNorm(group_count(c0), c0)
                self.out = nn.Conv2d(c0, 3, 3, padding=1)
                nn.init.zeros_(self.out.weight)
                nn.init.zeros_(self.out.bias)

            class ResidualBlock(nn.Module):
                def __init__(self, in_channels, out_channels):
                    super().__init__()
                    self.norm0 = nn.GroupNorm(group_count(in_channels), in_channels)
                    self.conv0 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
                    self.norm1 = nn.GroupNorm(group_count(out_channels), out_channels)
                    self.condition = nn.Linear(256, out_channels * 2)
                    self.conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
                    self.skip = (nn.Identity() if in_channels == out_channels
                                 else nn.Conv2d(in_channels, out_channels, 1))
                    nn.init.zeros_(self.conv1.weight)
                    nn.init.zeros_(self.conv1.bias)

                def forward(self, values, condition):
                    import torch.nn.functional as functional
                    hidden = self.conv0(functional.silu(self.norm0(values)))
                    scale, shift = self.condition(condition).chunk(2, dim=1)
                    hidden = self.norm1(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
                    hidden = self.conv1(functional.silu(hidden))
                    return (self.skip(values) + hidden) * (2 ** -0.5)

            class AttentionBlock(nn.Module):
                def __init__(self, channels, heads=4):
                    super().__init__()
                    if channels % heads:
                        raise ValueError("Attention channels must be divisible by heads")
                    self.channels = channels
                    self.heads = heads
                    self.norm = nn.GroupNorm(group_count(channels), channels)
                    self.qkv = nn.Conv2d(channels, channels * 3, 1)
                    self.projection = nn.Conv2d(channels, channels, 1)
                    self.scale = (channels // heads) ** -0.5
                    nn.init.zeros_(self.projection.weight)
                    nn.init.zeros_(self.projection.bias)

                def forward(self, values):
                    import torch
                    batch, channels, height, width = values.shape
                    q, k, v = self.qkv(self.norm(values)).reshape(
                        batch, 3, self.heads, channels // self.heads, height * width
                    ).unbind(1)
                    weights = torch.matmul(q.transpose(-2, -1) * self.scale, k).softmax(dim=-1)
                    hidden = torch.matmul(weights, v.transpose(-2, -1)).transpose(-2, -1)
                    hidden = hidden.reshape(batch, channels, height, width)
                    return values + self.projection(hidden)

            @staticmethod
            def time_embedding(values, width=128):
                import torch

                half = width // 2
                scale = math.log(10000) / max(half - 1, 1)
                frequencies = torch.exp(torch.arange(half, device=values.device) * -scale)
                angles = values.float()[:, None] * frequencies[None, :]
                return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

            @staticmethod
            def add_condition(values, projection, condition):
                return values + projection(condition)[:, :, None, None]

            def forward(self, values, timesteps, labels, structure_maps=None):
                import torch
                import torch.nn.functional as functional

                condition = self.time(self.time_embedding(timesteps)) + self.label(labels)
                h0 = self.in_conv(values)
                if self.structure_condition:
                    if structure_maps is None:
                        raise ValueError("Structure-conditioned diffusion requires target maps")
                    if structure_maps.ndim != 4 or structure_maps.shape[1] != 3:
                        raise ValueError(
                            f"Expected structure maps [N,3,H,W], got {tuple(structure_maps.shape)}"
                        )
                    maps = functional.interpolate(
                        structure_maps.to(dtype=h0.dtype),
                        size=values.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).clamp(0.0, 1.0)
                    h0 = h0 + self.structure_in(maps)
                for block in self.enc0:
                    h0 = block(h0, condition)
                h1 = self.down1(h0)
                for block in self.enc1:
                    h1 = block(h1, condition)
                h2 = self.down2(h1)
                for block in self.enc2:
                    h2 = block(h2, condition)
                h3 = self.down3(h2)
                for block in self.enc3:
                    h3 = block(h3, condition)
                middle = self.mid0(self.down4(h3), condition)
                middle = self.mid_attention(middle)
                middle = self.mid1(middle, condition)

                def decode(hidden, skip, upsample, block):
                    hidden = functional.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
                    hidden = upsample(hidden)
                    return block(torch.cat([hidden, skip], dim=1), condition)

                hidden = decode(middle, h3, self.up3, self.dec3)
                hidden = decode(hidden, h2, self.up2, self.dec2)
                hidden = decode(hidden, h1, self.up1, self.dec1)
                hidden = decode(hidden, h0, self.up0, self.dec0)
                return self.out(functional.silu(self.out_norm(hidden)))

        def group_count(channels):
            groups = min(8, channels)
            while channels % groups:
                groups -= 1
            return groups

        self.model = Model()


def diffusion_transform(image: Image.Image):
    import torch
    from torchvision import transforms

    return transforms.Compose([
        transforms.Lambda(lambda value: base.letterbox_square(value, GENERATOR_SIZE, pad_mode=base.GENERATOR_PAD_MODE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])(image)


def train_diffusion(train_frame: pd.DataFrame, seed: int) -> Path:
    import torch
    from torch.utils.data import DataLoader, Dataset

    base.seed_everything(seed)
    output = base.OUT / "synthetic_diffusion" / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    pool_per_class = int(os.environ.get("DIFFUSION_POOL_PER_CLASS", os.environ.get("GAN_POOL_PER_CLASS", "8192")))
    epochs = int(os.environ.get("DIFFUSION_EPOCHS", "1000"))
    timesteps = int(os.environ.get("DIFFUSION_TIMESTEPS", "100"))
    batch_size = int(os.environ.get("DIFFUSION_BATCH_SIZE", "4"))
    accumulation_steps = max(int(os.environ.get("DIFFUSION_ACCUMULATION_STEPS", "1")), 1)
    base_channels = int(os.environ.get("DIFFUSION_BASE_CHANNELS", "64"))
    learning_rate = float(os.environ.get("DIFFUSION_LR", "2e-4"))
    ema_decay = float(os.environ.get("DIFFUSION_EMA_DECAY", "0.999"))
    checkpoint_every = max(int(os.environ.get("DIFFUSION_CHECKPOINT_EVERY", "10")), 1)
    monitor_fraction = float(os.environ.get("DIFFUSION_MONITOR_FRACTION", "0.15"))
    min_epochs = max(int(os.environ.get("DIFFUSION_MIN_EPOCHS", "1")), 1)
    patience = max(int(os.environ.get("DIFFUSION_PATIENCE", "10")), 1)
    min_delta = float(os.environ.get("DIFFUSION_MIN_DELTA", "1e-4"))
    early_stop_monitor = os.environ.get("DIFFUSION_EARLY_STOP_MONITOR", "noise_mse").strip().lower()
    if early_stop_monitor not in {"noise_mse", "quality"}:
        raise ValueError("DIFFUSION_EARLY_STOP_MONITOR must be 'noise_mse' or 'quality'")
    schedule_name = os.environ.get("DIFFUSION_SCHEDULE", "cosine").strip().lower()
    quality_every = max(int(os.environ.get("DIFFUSION_QUALITY_EVERY", "5")), 1)
    quality_samples = max(int(os.environ.get("DIFFUSION_QUALITY_SAMPLES", "16")), 2)
    quality_sampling_steps = max(int(os.environ.get("DIFFUSION_QUALITY_SAMPLING_STEPS", "20")), 2)
    quality_sharpness_weight = float(os.environ.get("DIFFUSION_SHARPNESS_SELECTION_WEIGHT", "0.0"))
    min_snr_gamma = float(os.environ.get("DIFFUSION_MIN_SNR_GAMMA", "0.0"))
    structure_condition = os.environ.get("DIFFUSION_EYE_STRUCTURE_CONDITION", "0").strip().lower() in {
        "1", "true", "yes"
    }
    if not math.isfinite(min_snr_gamma) or min_snr_gamma < 0:
        raise ValueError("DIFFUSION_MIN_SNR_GAMMA must be finite and non-negative")
    perceptual_loss_weight = float(
        os.environ.get("DIFFUSION_PERCEPTUAL_LOSS_WEIGHT", "0.0")
    )
    if not math.isfinite(perceptual_loss_weight) or perceptual_loss_weight < 0:
        raise ValueError("DIFFUSION_PERCEPTUAL_LOSS_WEIGHT must be finite and non-negative")
    edge_structure_loss_weight = float(
        os.environ.get("DIFFUSION_EDGE_STRUCTURE_LOSS_WEIGHT", "0.0")
    )
    eye_structure_loss_weight = float(
        os.environ.get("DIFFUSION_EYE_STRUCTURE_LOSS_WEIGHT", "0.0")
    )
    eye_structure_selection_weight = float(
        os.environ.get("DIFFUSION_EYE_STRUCTURE_SELECTION_WEIGHT", "0.0")
    )
    eye_teacher_weights_value = os.environ.get("DIFFUSION_EYE_TEACHER_WEIGHTS", "").strip()
    eye_target_cache_value = os.environ.get("DIFFUSION_EYE_TARGET_CACHE", "").strip()
    if not math.isfinite(edge_structure_loss_weight) or edge_structure_loss_weight < 0:
        raise ValueError("DIFFUSION_EDGE_STRUCTURE_LOSS_WEIGHT must be finite and non-negative")
    if not math.isfinite(eye_structure_loss_weight) or eye_structure_loss_weight < 0:
        raise ValueError("DIFFUSION_EYE_STRUCTURE_LOSS_WEIGHT must be finite and non-negative")
    if not math.isfinite(eye_structure_selection_weight) or eye_structure_selection_weight < 0:
        raise ValueError("DIFFUSION_EYE_STRUCTURE_SELECTION_WEIGHT must be finite and non-negative")
    if not math.isfinite(quality_sharpness_weight) or quality_sharpness_weight < 0:
        raise ValueError("DIFFUSION_SHARPNESS_SELECTION_WEIGHT must be finite and non-negative")
    if not 0.05 <= monitor_fraction < 0.5:
        raise ValueError(f"DIFFUSION_MONITOR_FRACTION must be in [0.05, 0.5), got {monitor_fraction}")
    if base_channels < 16 or base_channels % 8:
        raise ValueError(f"DIFFUSION_BASE_CHANNELS must be a multiple of 8 and >=16, got {base_channels}")

    # Keep generator early stopping entirely inside the locked training cohort.
    # The monitor groups are never validation or test groups and are selected by
    # patient_group, so multiple photographs from one patient cannot straddle
    # the fitting and monitor subsets.
    from generator_quality import (split_generator_training, update_ema, ddim_step,
                                   DiffusionQualityMonitor)
    from eye_structure_prior import EyeTargetCache, PretrainedEyeTeacher
    fit_frame, monitor_frame = split_generator_training(train_frame, seed, monitor_fraction)
    if fit_frame.empty or monitor_frame.empty:
        raise RuntimeError("The diffusion fit/monitor split is empty")

    def ids_hash(frame: pd.DataFrame) -> str:
        return hashlib.sha256("\n".join(sorted(frame["image_id"].astype(str))).encode("utf-8")).hexdigest()

    eye_cache = None
    eye_reference = None
    eye_teacher_weights_path = None
    eye_teacher_weights_sha256 = None
    if eye_structure_loss_weight or eye_structure_selection_weight:
        if not eye_teacher_weights_value or not eye_target_cache_value:
            raise ValueError(
                "Eye-structure loss/selection requires DIFFUSION_EYE_TEACHER_WEIGHTS "
                "and DIFFUSION_EYE_TARGET_CACHE"
            )
        eye_teacher_weights_path = Path(eye_teacher_weights_value).resolve(strict=True)
        eye_teacher_weights_sha256 = base.sha256_file(eye_teacher_weights_path)
        eye_cache = EyeTargetCache(
            eye_target_cache_value,
            expected_weights_sha256=eye_teacher_weights_sha256,
        )
        missing_eye_targets = set(train_frame["image_id"].astype(str)) - set(eye_cache.index)
        if missing_eye_targets:
            raise ValueError(
                f"Eye target cache omits {len(missing_eye_targets)} training images; refusing partial alignment"
            )
        if eye_cache.metadata.get("image_ids_sha256") != ids_hash(train_frame):
            raise ValueError("Eye target cache image IDs do not match the active training cohort")
        eye_reference = eye_cache.reference_statistics(train_frame)
    if structure_condition and eye_cache is None:
        raise ValueError(
            "DIFFUSION_EYE_STRUCTURE_CONDITION requires the train-only eye target cache"
        )

    quality_metric_parts = ["class_mean_multiscale_spatial_polynomial_mmd2_resnet18_ddim"]
    if quality_sharpness_weight:
        quality_metric_parts.append("weighted_abs_log_sharpness_ratio_deviation")
    if eye_structure_selection_weight:
        quality_metric_parts.append("weighted_frozen_periorbital_eye_structure_penalty")

    config = {
        "seed": seed,
        "generator_type": "conditional_ddpm",
        "architecture": (
            "conditional_resunet_attention_16x16_structure_map"
            if structure_condition else "conditional_resunet_attention_16x16"
        ),
        "generator_size": GENERATOR_SIZE,
        "padding_mode": base.GENERATOR_PAD_MODE,
        "extreme_aspect_ratio": base.GENERATOR_EXTREME_ASPECT_RATIO,
        "pool_per_class": pool_per_class,
        "epochs": epochs,
        "timesteps": timesteps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "base_channels": base_channels,
        "structure_condition": structure_condition,
        "structure_condition_channels": 3 if structure_condition else 0,
        "structure_condition_source": (
            "train_eye_targets_iris_eyelid_trusted" if structure_condition else None
        ),
        "learning_rate": learning_rate,
        "ema_decay": ema_decay,
        "ema_warmup": "min(decay, (1+optimizer_updates)/(10+optimizer_updates))",
        "monitor_noise_seed": int(seed * 1000003 + 161803),
        "monitor_weights": "ema",
        "sampling_method": os.environ.get("DIFFUSION_SAMPLER", "ddpm"),
        "sampling_steps": int(os.environ.get("DIFFUSION_SAMPLING_STEPS", str(timesteps))),
        "quality_monitor": "patient_group_disjoint_train_monitor_multiscale_spatial_resnet18_mmd_ddim",
        "quality_selection_metric": " + ".join(quality_metric_parts),
        "quality_sharpness_selection_weight": quality_sharpness_weight,
        "min_snr_gamma": min_snr_gamma,
        "perceptual_loss_weight": perceptual_loss_weight,
        "edge_structure_loss_weight": edge_structure_loss_weight,
        "eye_structure_loss_weight": eye_structure_loss_weight,
        "eye_structure_selection_weight": eye_structure_selection_weight,
        "eye_structure_teacher_weights_sha256": eye_teacher_weights_sha256,
        "eye_structure_target_cache_sha256": (
            eye_cache.metadata.get("cache_sha256") if eye_cache is not None else None
        ),
        "eye_structure_target_cache_image_ids_sha256": (
            eye_cache.metadata.get("image_ids_sha256") if eye_cache is not None else None
        ),
        "quality_monitor_every": quality_every,
        "quality_monitor_samples_per_class": quality_samples,
        "quality_monitor_sampling_steps": quality_sampling_steps,
        "quality_code_sha256": base.sha256_file(BUNDLE / "generator_quality.py"),
        "checkpoint_every": checkpoint_every,
        "early_stopping": {
            "monitor": (
                "patient_group_disjoint_train_monitor_quality_selection_score"
                if early_stop_monitor == "quality"
                else "patient_group_disjoint_train_monitor_noise_mse"
            ),
            "monitor_fraction": monitor_fraction,
            "minimum_epochs": min_epochs,
            "patience": patience,
            "patience_unit": (
                "quality_monitor_evaluations" if early_stop_monitor == "quality" else "epochs"
            ),
            "quality_monitor_every": quality_every if early_stop_monitor == "quality" else None,
            "minimum_improvement": min_delta,
        },
        "noise_schedule": schedule_name,
        "train_image_ids_sha256": ids_hash(train_frame),
        "fit_image_ids_sha256": ids_hash(fit_frame),
        "monitor_image_ids_sha256": ids_hash(monitor_frame),
        "fit_patient_groups": int(fit_frame["patient_group"].nunique()),
        "monitor_patient_groups": int(monitor_frame["patient_group"].nunique()),
        "code_sha256": base.sha256_file(Path(__file__).resolve()),
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    if config["sampling_method"] not in {"ddpm", "ddim"} or not 2 <= config["sampling_steps"] <= timesteps:
        raise ValueError("Invalid diffusion sampling method or number of steps")
    marker = output / "diffusion_complete.json"
    if marker.is_file() and not base.FORCE_RERUN:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError(f"Existing diffusion output uses another data/code/settings fingerprint: {output}")
        if payload.get("configuration_fingerprint") == fingerprint and len(list(output.glob("pool_class*.png"))) == 2 * pool_per_class:
            return output
    device = torch.device(base.device_name())
    eye_teacher = None
    if eye_structure_loss_weight or eye_structure_selection_weight:
        eye_teacher = PretrainedEyeTeacher(eye_teacher_weights_path, device)

    class RealDataset(Dataset):
        def __init__(self, frame):
            self.frame = frame.reset_index(drop=True)

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, index):
            row = self.frame.iloc[index]
            with Image.open(row["absolute_path"]).convert("RGB") as image:
                value = diffusion_transform(image)
            if eye_cache is None:
                target = torch.zeros((3, 64, 64), dtype=torch.float32)
                active = torch.zeros((3,), dtype=torch.bool)
            else:
                target, active = eye_cache.get(str(row["image_id"]))
                if structure_condition:
                    # Missing teacher channels must not become a positive
                    # conditioning signal.  The original target and active
                    # gate are still returned for the distillation loss.
                    target = target * active[:, None, None].to(dtype=target.dtype)
            return value, torch.tensor(int(row["label"]), dtype=torch.long), target, active

    loader = DataLoader(RealDataset(fit_frame), batch_size=batch_size, shuffle=True, num_workers=base.NUM_WORKERS, drop_last=False)
    monitor_loader = DataLoader(RealDataset(monitor_frame), batch_size=batch_size, shuffle=False, num_workers=base.NUM_WORKERS, drop_last=False)
    model = DiffusionDenoiser(
        torch,
        base_channels=base_channels,
        timesteps=timesteps,
        structure_condition=structure_condition,
    ).model.to(device)

    def condition_maps(frame: pd.DataFrame, per_class: int | None = None) -> dict[int, torch.Tensor]:
        """Build deterministic teacher-map conditions from a train-only frame."""
        if not structure_condition:
            return {}
        if eye_cache is None:
            raise RuntimeError("Structure conditioning requested without an eye target cache")
        result: dict[int, torch.Tensor] = {}
        for label in (0, 1):
            rows = frame.loc[frame["label"].astype(int).eq(label)].sort_values("image_id")
            maps = []
            for _, row in rows.iterrows():
                target, active = eye_cache.get(str(row["image_id"]))
                target = target * active[:, None, None].to(dtype=target.dtype)
                # Only trusted iris+eyelid pairs are used as explicit anatomy
                # conditions, so the generator cannot learn from empty maps.
                if bool(active[1] and active[2]):
                    maps.append(target)
            if not maps:
                raise RuntimeError(f"No trusted structure maps for conditioning class {label}")
            if per_class is not None:
                order = np.arange(len(maps), dtype=int)
                rng = np.random.default_rng(seed + 7919 + label * 104729)
                rng.shuffle(order)
                maps = [maps[int(order[index % len(order)])] for index in range(per_class)]
            result[label] = torch.stack(maps, dim=0)
        return result

    fit_condition_pool = {
        label: maps.to(device) for label, maps in condition_maps(fit_frame).items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if schedule_name == "cosine":
        schedule_steps = torch.arange(timesteps + 1, device=device, dtype=torch.float32)
        schedule_offset = 0.008
        schedule_x = (schedule_steps / timesteps + schedule_offset) / (1.0 + schedule_offset)
        schedule_bar = torch.cos(schedule_x * math.pi / 2.0).square()
        schedule_bar = schedule_bar / schedule_bar[0]
        beta = (1.0 - schedule_bar[1:] / schedule_bar[:-1]).clamp(1e-5, 0.999)
    elif schedule_name == "linear":
        beta = torch.linspace(1e-4, 0.02, timesteps, device=device)
    else:
        raise ValueError(f"Unsupported DIFFUSION_SCHEDULE={schedule_name!r}; use cosine or linear")
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    alpha_bar_previous = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]])
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
    snr = alpha_bar / (1.0 - alpha_bar).clamp_min(1e-8)
    posterior_variance = (beta * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar)).clamp_min(1e-20)
    posterior_mean_coef1 = beta * torch.sqrt(alpha_bar_previous) / (1.0 - alpha_bar)
    posterior_mean_coef2 = (1.0 - alpha_bar_previous) * torch.sqrt(alpha) / (1.0 - alpha_bar)

    def sobel_edge_grid(values):
        """Return a low-resolution central Sobel map while avoiding padded borders."""
        import torch.nn.functional as functional

        gray = (
            values[:, 0:1] * 0.299
            + values[:, 1:2] * 0.587
            + values[:, 2:3] * 0.114
        )
        kernel_x = values.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None] / 8.0
        kernel_y = values.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])[None, None] / 8.0
        padded = functional.pad(gray, (1, 1, 1, 1), mode="reflect")
        edges = torch.sqrt(
            functional.conv2d(padded, kernel_x).square()
            + functional.conv2d(padded, kernel_y).square()
            + 1e-6
        )
        margin = max(2, GENERATOR_SIZE // 16)
        if edges.shape[-1] > 2 * margin:
            edges = edges[:, :, margin:-margin, margin:-margin]
        return functional.adaptive_avg_pool2d(edges, (16, 16))

    def diffusion_edge_structure_loss(clean, noisy, predicted_noise, steps):
        """Guide x0 estimates toward real eyelid/iris edge layouts at usable SNRs."""
        import torch.nn.functional as functional

        x0_hat = (
            noisy - sqrt_one_minus_alpha_bar[steps, None, None, None] * predicted_noise
        ) / sqrt_alpha_bar[steps, None, None, None].clamp_min(1e-4)
        x0_hat = x0_hat.clamp(-1.0, 1.0)
        predicted_edges = sobel_edge_grid(x0_hat)
        with torch.no_grad():
            clean_edges = sobel_edge_grid(clean)
        per_sample = functional.smooth_l1_loss(
            predicted_edges, clean_edges, reduction="none"
        ).mean(dim=(1, 2, 3))
        # The estimate is reliable mainly at lower noise levels; damp the
        # edge term smoothly as alpha_bar approaches zero.
        snr_weight = sqrt_alpha_bar[steps].detach().clamp_min(0.05)
        return (per_sample * snr_weight).mean()

    def diffusion_perceptual_loss(clean, noisy, predicted_noise, steps):
        """Match frozen multiscale local features of low-noise x0 estimates."""
        import torch.nn.functional as functional

        x0_hat = (
            noisy - sqrt_one_minus_alpha_bar[steps, None, None, None] * predicted_noise
        ) / sqrt_alpha_bar[steps, None, None, None].clamp_min(1e-4)
        x0_hat = x0_hat.clamp(-1.0, 1.0)
        predicted_features = perceptual_features(x0_hat)
        with torch.no_grad():
            clean_features = perceptual_features(clean)
        per_sample = functional.smooth_l1_loss(
            predicted_features, clean_features, reduction="none"
        ).mean(dim=1)
        snr_weight = sqrt_alpha_bar[steps].detach().clamp_min(0.05)
        return (per_sample * snr_weight).mean()

    checkpoint_path = output / "diffusion_checkpoint.pt"
    best_state, best_ema_state, best_monitor_loss = None, None, np.inf
    best_quality_ema_state, best_quality_score, best_quality_epoch = None, np.inf, 0
    best_quality_mmd2 = np.inf
    best_quality_eye_structure_penalty = np.inf
    stale = 0
    best_epoch = 0
    history = []
    optimizer_updates = 0
    pd.concat([fit_frame.assign(generator_partition="fit"),
               monitor_frame.assign(generator_partition="monitor")]).to_csv(output / "diffusion_training_partition.csv", index=False)
    ema_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    }
    start_epoch = 1
    if checkpoint_path.is_file() and not base.FORCE_RERUN:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError(f"Diffusion checkpoint does not match this run; use a new output root: {checkpoint_path}")
        if checkpoint.get("configuration_fingerprint") == fingerprint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            ema_state = {
                key: value.to(device)
                for key, value in checkpoint.get("ema_state", {}).items()
            }
            best_state = checkpoint.get("best_state")
            best_ema_state = checkpoint.get("best_ema_state")
            best_quality_ema_state = checkpoint.get("best_quality_ema_state")
            best_quality_score = float(checkpoint.get("best_quality_score", np.inf))
            best_quality_epoch = int(checkpoint.get("best_quality_epoch", 0))
            best_quality_mmd2 = float(checkpoint.get("best_quality_mmd2", best_quality_score))
            best_quality_eye_structure_penalty = float(
                checkpoint.get("best_quality_eye_structure_penalty", np.inf)
            )
            best_monitor_loss = float(checkpoint.get("best_monitor_loss", checkpoint.get("best_loss", np.inf)))
            best_epoch = int(checkpoint.get("best_epoch", 0))
            stale = int(checkpoint.get("stale", 0))
            history = list(checkpoint.get("history", []))
            optimizer_updates = int(checkpoint.get("optimizer_updates", 0))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            # Checkpoints are loaded onto the active device, but RNG restore
            # requires CPU uint8 state tensors. Normalize both forms so a run
            # can resume on a different CUDA device without failing here.
            def _cpu_rng_state(value):
                if value is None:
                    return None
                if isinstance(value, torch.Tensor):
                    return value.detach().to(device="cpu", dtype=torch.uint8)
                return torch.as_tensor(value, dtype=torch.uint8, device="cpu")

            rng_state = _cpu_rng_state(checkpoint.get("rng_state"))
            if rng_state is not None:
                torch.set_rng_state(rng_state)
            cuda_rng_state = checkpoint.get("cuda_rng_state")
            if torch.cuda.is_available() and cuda_rng_state is not None:
                if isinstance(cuda_rng_state, torch.Tensor):
                    cuda_rng_state = [cuda_rng_state]
                normalized_cuda_states = [
                    _cpu_rng_state(value) for value in cuda_rng_state
                ]
                torch.cuda.set_rng_state_all(normalized_cuda_states)

    monitor_condition_maps = condition_maps(
        monitor_frame,
        per_class=config["quality_monitor_samples_per_class"],
    )
    quality_monitor = DiffusionQualityMonitor(
        monitor_frame, diffusion_transform, device, seed, alpha_bar,
        image_size=GENERATOR_SIZE,
        per_class=config["quality_monitor_samples_per_class"],
        sampling_steps=config["quality_monitor_sampling_steps"],
        structure_conditions=monitor_condition_maps if structure_condition else None,
    )

    def perceptual_features(values):
        """Differentiable counterpart of the frozen quality encoder features."""
        import torch.nn.functional as functional

        values = ((values + 1) / 2).clamp(0, 1)
        values = functional.interpolate(
            values, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
        )
        mean = values.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = values.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        x = (values - mean) / std
        encoder = quality_monitor.encoder
        x = encoder.relu(encoder.bn1(encoder.conv1(x)))
        x = encoder.maxpool(x)
        outputs = []
        for layer, spatial_size in (
            (encoder.layer1, 4),
            (encoder.layer2, 2),
            (encoder.layer3, 1),
            (encoder.layer4, 1),
        ):
            x = layer(x)
            pooled = functional.adaptive_avg_pool2d(x, (spatial_size, spatial_size))
            outputs.append(functional.normalize(pooled.flatten(1), dim=1))
        return torch.cat(outputs, dim=1) / math.sqrt(4.0)

    def monitor_noise_mse(epoch: int) -> float:
        model.eval()
        generator = torch.Generator(device=device.type)
        generator.manual_seed(config["monitor_noise_seed"])
        total, samples = 0.0, 0
        with torch.no_grad():
            for values, labels, eye_targets, _ in monitor_loader:
                values, labels = values.to(device), labels.to(device)
                eye_targets = eye_targets.to(device)
                steps = torch.randint(0, timesteps, (len(values),), generator=generator, device=device)
                noise = torch.randn(values.shape, generator=generator, device=device, dtype=values.dtype)
                noisy = sqrt_alpha_bar[steps, None, None, None] * values + sqrt_one_minus_alpha_bar[steps, None, None, None] * noise
                arguments = (noisy, steps, labels)
                if structure_condition:
                    arguments = arguments + (eye_targets,)
                prediction = torch.func.functional_call(model, ema_state, arguments)
                total += float(torch.nn.functional.mse_loss(prediction, noise).item()) * len(values)
                samples += len(values)
        return total / max(samples, 1)

    if start_epoch > min_epochs and stale >= patience:
        start_epoch = epochs + 1
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_noise_mse, total_weighted_noise_mse = 0.0, 0.0
        total_edge_loss, total_perceptual_loss, total_eye_structure_loss, batches = 0.0, 0.0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (values, labels, eye_targets, eye_active) in enumerate(loader):
            values, labels = values.to(device), labels.to(device)
            eye_targets = eye_targets.to(device)
            steps = torch.randint(0, timesteps, (len(values),), device=device)
            noise = torch.randn_like(values)
            noisy = sqrt_alpha_bar[steps, None, None, None] * values + sqrt_one_minus_alpha_bar[steps, None, None, None] * noise
            model_arguments = (noisy, steps, labels)
            if structure_condition:
                model_arguments = model_arguments + (eye_targets,)
            prediction = model(*model_arguments)
            noise_mse_per_sample = torch.nn.functional.mse_loss(
                prediction, noise, reduction="none"
            ).mean(dim=(1, 2, 3))
            noise_mse = noise_mse_per_sample.mean()
            if min_snr_gamma:
                # Min-SNR-gamma for epsilon prediction.  The unclipped
                # objective remains the logged/monitoring metric; only the
                # optimization term is reweighted by timestep SNR.
                snr_weights = torch.minimum(
                    snr[steps],
                    torch.full_like(steps, min_snr_gamma, dtype=torch.float32),
                ) / snr[steps].clamp_min(1e-8)
                weighted_noise_mse = (noise_mse_per_sample * snr_weights).mean()
            else:
                weighted_noise_mse = noise_mse
            edge_loss = torch.zeros((), device=device)
            if edge_structure_loss_weight:
                edge_loss = diffusion_edge_structure_loss(values, noisy, prediction, steps)
            perceptual_loss = torch.zeros((), device=device)
            if perceptual_loss_weight:
                perceptual_loss = diffusion_perceptual_loss(values, noisy, prediction, steps)
            eye_structure_loss = torch.zeros((), device=device)
            if eye_structure_loss_weight:
                predicted_clean = (
                    noisy - sqrt_one_minus_alpha_bar[steps, None, None, None] * prediction
                ) / sqrt_alpha_bar[steps, None, None, None].clamp_min(1e-4)
                eye_structure_loss = eye_teacher.loss(
                    predicted_clean,
                    eye_targets.to(device),
                    eye_active.to(device),
                    sqrt_alpha_bar[steps].detach().clamp_min(0.05),
                )
            loss = (
                weighted_noise_mse
                + edge_structure_loss_weight * edge_loss
                + perceptual_loss_weight * perceptual_loss
                + eye_structure_loss_weight * eye_structure_loss
            )
            bucket_start = (batch_index // accumulation_steps) * accumulation_steps
            bucket_steps = min(accumulation_steps, len(loader) - bucket_start)
            (loss / bucket_steps).backward()
            if (batch_index + 1) % accumulation_steps == 0 or (batch_index + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
                update_ema(ema_state, model, ema_decay, optimizer_updates)
            total_noise_mse += float(noise_mse.item())
            total_weighted_noise_mse += float(weighted_noise_mse.detach().item())
            total_edge_loss += float(edge_loss.detach().item())
            total_perceptual_loss += float(perceptual_loss.detach().item())
            total_eye_structure_loss += float(eye_structure_loss.detach().item())
            batches += 1
        mean_noise_mse = total_noise_mse / max(batches, 1)
        mean_weighted_noise_mse = total_weighted_noise_mse / max(batches, 1)
        mean_edge_loss = total_edge_loss / max(batches, 1)
        mean_perceptual_loss = total_perceptual_loss / max(batches, 1)
        mean_eye_structure_loss = total_eye_structure_loss / max(batches, 1)
        mean_objective = (
            mean_weighted_noise_mse
            + edge_structure_loss_weight * mean_edge_loss
            + perceptual_loss_weight * mean_perceptual_loss
            + eye_structure_loss_weight * mean_eye_structure_loss
        )
        monitor_loss = monitor_noise_mse(epoch)
        if not np.isfinite(monitor_loss):
            raise FloatingPointError("Non-finite diffusion EMA monitor loss")
        if monitor_loss < best_monitor_loss - min_delta:
            best_monitor_loss = monitor_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_ema_state = {key: value.detach().cpu().clone() for key, value in ema_state.items()}
            if early_stop_monitor == "noise_mse":
                stale = 0
        elif early_stop_monitor == "noise_mse":
            stale += 1
        quality_mmd2, quality_details, quality_preview = np.nan, {}, None
        quality_selection_score = np.nan
        quality_sharpness_log_distance = np.nan
        quality_sharpness_penalty = np.nan
        quality_eye_structure_penalty = np.nan
        quality_eye_structure_selection_penalty = np.nan
        if epoch % config["quality_monitor_every"] == 0 or epoch == min_epochs:
            quality_mmd2, quality_details, quality_preview = quality_monitor.evaluate(model, ema_state)
            class_sharpness_log_distances = [
                abs(math.log(
                    max(quality_details[f"sharpness_generated_class_{label}"], 1e-12)
                    / max(quality_details[f"sharpness_real_class_{label}"], 1e-12)
                ))
                for label in (0, 1)
            ]
            quality_sharpness_log_distance = float(np.mean(class_sharpness_log_distances))
            quality_sharpness_penalty = quality_sharpness_weight * quality_sharpness_log_distance
            if eye_teacher is not None and eye_structure_selection_weight:
                eye_scores = []
                for label in (0, 1):
                    eye_score, eye_details = eye_teacher.generated_quality(
                        quality_monitor.last_generated[label].to(device),
                        eye_reference[label],
                    )
                    eye_scores.append(eye_score)
                    quality_details.update({
                        f"quality_class_{label}_{key}": value
                        for key, value in eye_details.items()
                    })
                quality_eye_structure_penalty = float(np.mean(eye_scores))
                quality_eye_structure_selection_penalty = (
                    eye_structure_selection_weight * quality_eye_structure_penalty
                )
            else:
                quality_eye_structure_penalty = 0.0
                quality_eye_structure_selection_penalty = 0.0
            quality_selection_score = float(
                quality_mmd2 + quality_sharpness_penalty + quality_eye_structure_selection_penalty
            )
            quality_details["quality_mmd2_raw"] = float(quality_mmd2)
            quality_details["quality_sharpness_log_distance"] = quality_sharpness_log_distance
            quality_details["quality_sharpness_selection_penalty"] = quality_sharpness_penalty
            quality_details["quality_eye_structure_penalty"] = quality_eye_structure_penalty
            quality_details["quality_eye_structure_selection_penalty"] = quality_eye_structure_selection_penalty
            if quality_selection_score < best_quality_score - min_delta:
                best_quality_score = quality_selection_score
                best_quality_mmd2 = float(quality_mmd2)
                best_quality_eye_structure_penalty = float(quality_eye_structure_penalty)
                best_quality_epoch = epoch
                best_quality_ema_state = {key: value.detach().cpu().clone() for key, value in ema_state.items()}
                if early_stop_monitor == "quality":
                    stale = 0
                from torchvision.utils import save_image
                save_image((quality_preview + 1) / 2, output / "best_fixed_examples.png", nrow=4)
            elif early_stop_monitor == "quality":
                stale += 1
        history.append({
            "epoch": epoch,
            "train_noise_mse": mean_noise_mse,
            "train_weighted_noise_mse": mean_weighted_noise_mse,
            "train_edge_structure_loss": mean_edge_loss,
            "train_perceptual_loss": mean_perceptual_loss,
            "train_eye_structure_loss": mean_eye_structure_loss,
            "train_objective": mean_objective,
            "train_monitor_noise_mse": monitor_loss,
            "best_train_monitor_noise_mse": best_monitor_loss,
            "early_stop_stale_count": stale,
            "early_stopping_patience_unit": config["early_stopping"]["patience_unit"],
            "ema_decay": ema_decay,
            "quality_mmd2": quality_mmd2,
            "best_quality_mmd2": best_quality_mmd2,
            "quality_selection_score": quality_selection_score,
            "best_quality_selection_score": best_quality_score,
            "quality_sharpness_log_distance": quality_sharpness_log_distance,
            "quality_sharpness_selection_penalty": quality_sharpness_penalty,
            "quality_eye_structure_penalty": quality_eye_structure_penalty,
            "quality_eye_structure_selection_penalty": quality_eye_structure_selection_penalty,
            "quality_monitor_epoch": best_quality_epoch,
            **quality_details,
        })
        print(
            f"[DIFFUSION] seed={seed} epoch={epoch}/{epochs} "
            f"fit_mse={mean_noise_mse:.6f} weighted={mean_weighted_noise_mse:.6f} "
            f"edge={mean_edge_loss:.6f} perceptual={mean_perceptual_loss:.6f} "
            f"eye={mean_eye_structure_loss:.6f} "
            f"monitor_mse={monitor_loss:.6f} "
            f"best={best_monitor_loss:.6f} stale={stale}/{patience} "
            f"unit={config['early_stopping']['patience_unit']}"
        )
        if epoch % checkpoint_every == 0 or epoch == epochs or (epoch >= min_epochs and stale >= patience):
            temporary_checkpoint = checkpoint_path.with_suffix(".tmp")
            torch.save({
                "configuration_fingerprint": fingerprint,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema_state": {key: value.detach().cpu() for key, value in ema_state.items()},
                "best_state": best_state,
                "best_ema_state": best_ema_state,
                "best_monitor_loss": best_monitor_loss,
                "best_quality_ema_state": best_quality_ema_state,
                "best_quality_score": best_quality_score,
                "best_quality_mmd2": best_quality_mmd2,
                "best_quality_eye_structure_penalty": best_quality_eye_structure_penalty,
                "best_quality_epoch": best_quality_epoch,
                "best_epoch": best_epoch,
                "stale": stale,
                "optimizer_updates": optimizer_updates,
                "history": history,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }, temporary_checkpoint)
            temporary_checkpoint.replace(checkpoint_path)
        if epoch >= min_epochs and stale >= patience:
            print(f"[DIFFUSION EARLY STOP] seed={seed} epoch={epoch} after {stale} stale epochs")
            break
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_ema_state is None:
        best_ema_state = {key: value.detach().cpu().clone() for key, value in ema_state.items()}
    selected_state = dict(best_state)
    selected_state.update(best_quality_ema_state or best_ema_state)
    model.load_state_dict(selected_state, strict=False)
    model.eval()
    # A new protocol uses its own output root; do not delete a prior image pool.
    sampling_steps = np.unique(np.linspace(0, timesteps - 1, config["sampling_steps"], dtype=int))[::-1]
    with torch.no_grad():
        condition_rng = torch.Generator(device=device.type).manual_seed(seed + 424242)
        for label in [0, 1]:
            generated = 0
            while generated < pool_per_class:
                count = min(int(os.environ.get("DIFFUSION_GENERATION_BATCH", "4")), pool_per_class - generated)
                values = torch.randn(count, 3, GENERATOR_SIZE, GENERATOR_SIZE, device=device)
                labels = torch.full((count,), label, dtype=torch.long, device=device)
                if structure_condition:
                    pool = fit_condition_pool[label]
                    indices = torch.randint(
                        len(pool), (count,), generator=condition_rng, device=device
                    )
                    generation_conditions = pool[indices]
                else:
                    generation_conditions = None
                reverse_steps = sampling_steps if config["sampling_method"] == "ddim" else list(reversed(range(timesteps)))
                for step_index, step in enumerate(reverse_steps):
                    current = torch.full((count,), step, dtype=torch.long, device=device)
                    model_arguments = (values, current, labels)
                    if structure_condition:
                        model_arguments = model_arguments + (generation_conditions,)
                    predicted_noise = model(*model_arguments)
                    if config["sampling_method"] == "ddim":
                        previous = reverse_steps[step_index + 1] if step_index + 1 < len(reverse_steps) else -1
                        previous_alpha = alpha_bar[previous] if previous >= 0 else values.new_tensor(1.0)
                        values = ddim_step(values, predicted_noise, alpha_bar[step], previous_alpha)
                        continue
                    alpha_bar_value = alpha_bar[step]
                    x0 = (values - sqrt_one_minus_alpha_bar[step] * predicted_noise) / torch.sqrt(alpha_bar_value)
                    x0 = x0.clamp(-1.0, 1.0)
                    values = posterior_mean_coef1[step] * x0 + posterior_mean_coef2[step] * values
                    if step > 0:
                        values = values + torch.sqrt(posterior_variance[step]) * torch.randn_like(values)
                values = ((values.clamp(-1, 1) + 1.0) * 127.5).byte().cpu()
                for offset, value in enumerate(values):
                    Image.fromarray(value.permute(1, 2, 0).numpy()).save(output / f"pool_class{label}_{generated + offset:05d}.png")
                generated += count
    pd.DataFrame(history).to_csv(output / "diffusion_history.csv", index=False)
    marker.write_text(json.dumps({
        **config,
        "configuration_fingerprint": fingerprint,
        "best_train_monitor_noise_mse": float(best_monitor_loss),
        "best_quality_mmd2": float(best_quality_mmd2),
        "best_quality_selection_score": float(best_quality_score),
        "best_quality_eye_structure_penalty": float(best_quality_eye_structure_penalty),
        "quality_sharpness_selection_weight": quality_sharpness_weight,
        "best_quality_epoch": int(best_quality_epoch),
        "best_epoch": int(best_epoch),
        "completed_epochs": int(history[-1]["epoch"]) if history else 0,
        "stopped_early": bool(history and history[-1]["epoch"] < epochs),
        "early_stopping_monitor": config["early_stopping"]["monitor"],
        "early_stopping_patience_unit": config["early_stopping"]["patience_unit"],
        "selected_checkpoint": "ema",
        "checkpoint_file": checkpoint_path.name,
        "status": "complete",
    }, indent=2), encoding="utf-8")
    return output


def make_synthetic_frame(pool_dir: Path | None, real_frame: pd.DataFrame, ratio: float, strategy: str, seed: int, method: str) -> pd.DataFrame:
    if ratio <= 0:
        return pd.DataFrame(columns=["image_id", "absolute_path", "label", "label_name", "patient_group", "synthetic"])
    if pool_dir is None:
        raise RuntimeError(f"Synthetic ratio n={ratio:g} requires a completed {method} pool")
    total = int(round(ratio * len(real_frame)))
    counts = real_frame["label"].value_counts().to_dict()
    if strategy == "balanced":
        if (len(real_frame) + total) % 2:
            total += 1
        benign = int(counts.get(0, 0))
        malignant = int(counts.get(1, 0))
        benign_synthetic = (total - benign + malignant) // 2
        target_counts = {0: benign_synthetic, 1: total - benign_synthetic}
    else:
        target_counts = {label: int(round(total * counts.get(label, 0) / max(len(real_frame), 1))) for label in [0, 1]}
        target_counts[1] += total - sum(target_counts.values())
    rows = []
    for label, count in target_counts.items():
        candidates = sorted(pool_dir.glob(f"pool_class{label}_*.png"))
        if count > len(candidates):
            raise RuntimeError(f"{method} pool class {label} has {len(candidates)} images but route requires {count}")
        rng = np.random.default_rng(seed + 104729 * label)
        ordered = np.asarray(candidates, dtype=object)[rng.permutation(len(candidates))]
        for rank, path in enumerate(ordered[:count]):
            rows.append({
                "image_id": f"syn_{method}_{seed}_{strategy}_{ratio}_{label}_{rank:05d}",
                "absolute_path": str(path),
                "label": int(label),
                "label_name": "malignant" if label else "benign",
                "patient_group": f"synthetic_{method}_{seed}_{strategy}_{ratio}_{label}_{rank:05d}",
                "synthetic": True,
                "synthetic_pool_rank": int(rank),
            })
    return pd.DataFrame(rows)


def fit_head(head: str, train_x, train_y, val_x, val_y, seed: int):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC

    normalized = head.lower().replace("-", "_")
    if normalized == "naive_bayes":
        model = GaussianNB()
        model.fit(train_x, train_y)
        return model
    if normalized == "svm":
        model = SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
        model.fit(train_x, train_y)
        return model
    if normalized == "random_forest":
        return base.fit_random_forest(train_x, train_y, seed)
    if normalized == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise RuntimeError("xgboost is required for the full matrix") from exc
        params = {
            "learning_rate": 0.005,
            "n_estimators": 2000,
            "early_stopping_rounds": 75,
            "max_depth": 4,
            "min_child_weight": 5,
            "gamma": 0.6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": seed,
            "n_jobs": 4,
            "tree_method": "hist",
        }
        if base.device_name() == "cuda":
            params["device"] = "cuda"
        model = XGBClassifier(**params)
        model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
        model._sycnn_device = params.get("device", "cpu")
        model._sycnn_best_iteration = getattr(model, "best_iteration", None)
        model._sycnn_best_score = getattr(model, "best_score", None)
        return model
    raise ValueError(f"Unsupported full-matrix head: {head}")


def _xgboost_probabilities(model, values: np.ndarray) -> np.ndarray:
    """Predict with an explicit device-compatible XGBoost input path.

    XGBoost boosters trained with ``device='cuda'`` warn and fall back when
    given a CPU NumPy array through ``XGBClassifier.predict_proba``.  Prefer
    a CuPy CUDA array when CuPy is available; otherwise construct an explicit
    DMatrix so the fallback is intentional, visible, and uses the booster
    prediction path rather than implicit inplace-predict negotiation.
    """
    import numpy as np
    import xgboost as xgb

    values = np.asarray(values, dtype=np.float32, order="C")
    best_iteration = getattr(model, "_sycnn_best_iteration", None)
    if best_iteration is None:
        best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        iteration_range = (0, 0)
    else:
        iteration_range = (0, int(best_iteration) + 1)

    device = getattr(model, "_sycnn_device", None)
    if device is None:
        try:
            device = model.get_xgb_params().get("device", "cpu")
        except Exception:
            device = "cpu"
    device = str(device)
    booster = model.get_booster()
    if device.startswith("cuda"):
        try:
            import cupy as cp
        except ImportError:
            cp = None
        if cp is not None:
            cuda_values = cp.asarray(values)
            predicted = booster.inplace_predict(
                cuda_values,
                iteration_range=iteration_range,
                validate_features=False,
            )
            model._sycnn_prediction_backend = "cupy_inplace_cuda"
            return cp.asnumpy(predicted).astype(np.float32, copy=False)

    matrix = xgb.DMatrix(values)
    predicted = booster.predict(matrix, iteration_range=iteration_range)
    model._sycnn_prediction_backend = "explicit_dmatrix"
    return np.asarray(predicted, dtype=np.float32)


def _head_probabilities(head: str, model, values: np.ndarray) -> np.ndarray:
    if head.lower().replace("-", "_") == "xgboost":
        return _xgboost_probabilities(model, values)
    return model.predict_proba(values)[:, 1]


def head_probabilities(head: str, model, model_probabilities: dict[str, np.ndarray], features: dict[str, np.ndarray]) -> np.ndarray:
    if head == "cnn":
        return model_probabilities["current"]
    return _head_probabilities(head, model, features["current"])


def save_generation_quality_audit(train_frame: pd.DataFrame, pool_dir: Path, method: str, seed: int) -> None:
    """Write class-conditional feature-space fidelity and diversity evidence."""
    output = pool_dir / "generation_quality_metrics.csv"
    marker = pool_dir / "generation_quality_complete.json"
    generation_marker = pool_dir / f"{method}_complete.json"
    generation_fingerprint = None
    pool_count = len(list(pool_dir.glob("pool_class*.png")))
    if generation_marker.is_file():
        try:
            generation_fingerprint = json.loads(
                generation_marker.read_text(encoding="utf-8")
            ).get("configuration_fingerprint")
        except (OSError, json.JSONDecodeError):
            generation_fingerprint = None
    if not generation_fingerprint:
        raise RuntimeError(
            f"Cannot audit {method} pool without a completed generator fingerprint: {generation_marker}"
        )
    if output.is_file() and marker.is_file() and not base.FORCE_RERUN:
        try:
            cached = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if (
            cached.get("source_generation_fingerprint") == generation_fingerprint
            and cached.get("n_pool_images") == pool_count
            and cached.get("metric_version") == 3
            and cached.get("audit_code_sha256") == base.sha256_file(Path(__file__).resolve())
        ):
            return
    sample_size = int(os.environ.get("GENERATION_QUALITY_SAMPLE", "512"))
    import torch
    from torchvision import transforms
    from generator_quality import GanQualityMonitor, polynomial_mmd2
    transform = transforms.Compose([
        transforms.Lambda(lambda image: base.letterbox_square(image, GENERATOR_SIZE, pad_mode=base.GENERATOR_PAD_MODE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    monitor = GanQualityMonitor(train_frame, transform, base.device_name(), seed,
                                per_class=min(sample_size, 128))
    rows = []
    panel_rows = []
    for label in [0, 1]:
        synthetic_paths = sorted(pool_dir.glob(f"pool_class{label}_*.png"))
        if not synthetic_paths:
            continue
        rng = np.random.default_rng(seed + label * 32452843)
        selected = list(np.asarray(synthetic_paths, dtype=object)[rng.permutation(len(synthetic_paths))[:sample_size]])
        tensors = []
        for path in selected:
            with Image.open(path).convert("RGB") as image:
                tensors.append(transform(image))
        synthetic_values = torch.stack(tensors).to(monitor.device)
        panel_rows.extend([monitor.real_images[label], synthetic_values[:4].cpu()])
        features = monitor.features(synthetic_values)
        mmd = float(polynomial_mmd2(monitor.real[label], features))
        normalized = torch.nn.functional.normalize(features, dim=1)
        diversity = float(torch.pdist(normalized).mean()) if len(features) > 1 else float("nan")
        real_sharpness = monitor.real_sharpness[label]
        generated_sharpness = float(monitor.sharpness(synthetic_values).mean())
        for metric, value in [("class_conditional_polynomial_mmd2", mmd),
                              ("synthetic_feature_diversity_mean_pairwise_distance", diversity),
                              ("real_laplacian_sharpness_mean", real_sharpness),
                              ("synthetic_laplacian_sharpness_mean", generated_sharpness),
                              ("synthetic_to_real_sharpness_ratio", generated_sharpness / max(real_sharpness, 1e-12))]:
            rows.append({"seed": seed, "generation_method": method, "label": label,
                         "label_name": "malignant" if label else "benign", "metric": metric,
                          "value": value, "feature_space": "ImageNet_ResNet18_multiscale_spatial",
                         "n_real_sampled": len(monitor.real[label]), "n_synthetic_sampled": len(features)})
    pd.DataFrame(rows).to_csv(output, index=False)
    if panel_rows:
        from torchvision.utils import save_image
        save_image((torch.cat(panel_rows) + 1) / 2, pool_dir / "final_pool_quality_examples.png", nrow=4)
    marker.write_text(json.dumps({
        "status": "complete",
        "generation_method": method,
        "source_generation_fingerprint": generation_fingerprint,
        "n_pool_images": pool_count,
        "input_size": IMAGE_SIZE,
         "metric_version": 4,
        "audit_code_sha256": base.sha256_file(Path(__file__).resolve()),
         "feature_space": "ImageNet_ResNet18_multiscale_spatial",
        "sample_size_per_class": sample_size,
        "example_panel_rows": ["benign_real", "benign_synthetic", "malignant_real", "malignant_synthetic"],
    }, indent=2), encoding="utf-8")


def save_similarity_audit(real_frame: pd.DataFrame, synthetic: pd.DataFrame, route_dir: Path, seed: int, method: str) -> None:
    if synthetic.empty:
        return
    output = route_dir / "similarity"
    output.mkdir(parents=True, exist_ok=True)
    unique = synthetic.drop_duplicates("absolute_path").copy()
    real = real_frame.copy()
    real_arrays, real_embeddings = [], []
    pad_mode = "reflect" if os.environ.get("MANUAL_EYE_DATASET_ROOT") else "constant"

    def embedding(array):
        compact = array.resize((32, 32), Image.Resampling.BILINEAR)
        value = np.asarray(compact, dtype=np.float32).reshape(-1) / 255.0
        gx = np.diff(value.reshape(32, 32), axis=1, prepend=value.reshape(32, 32)[:, :1]).reshape(-1)
        gy = np.diff(value.reshape(32, 32), axis=0, prepend=value.reshape(32, 32)[:1, :]).reshape(-1)
        vector = np.concatenate([value, gx, gy])
        return vector / max(float(np.linalg.norm(vector)), 1e-8)

    for row in real.itertuples(index=False):
        with Image.open(row.absolute_path).convert("L") as image:
            padded = base.letterbox_square(image.convert("RGB"), IMAGE_SIZE, pad_mode=pad_mode).convert("L")
            real_arrays.append(np.asarray(padded, dtype=np.float32) / 255.0)
            real_embeddings.append(embedding(padded))
    real_embeddings = np.asarray(real_embeddings)
    real_dhash = [int(str(value), 16) for value in real["dhash"]]
    real_md5 = real.groupby("md5")["image_id"].apply(list).to_dict()
    duplicate_rows, self_rows = [], []
    for row in unique.itertuples(index=False):
        path = Path(row.absolute_path)
        synthetic_md5 = base.md5(path)
        synthetic_dhash = base.dhash(path)
        hamming = np.asarray([(int(synthetic_dhash, 16) ^ value).bit_count() for value in real_dhash])
        nearest = int(np.argmin(hamming))
        duplicate_rows.append({
            "seed": seed,
            "generation_method": method,
            "synthetic_id": row.image_id,
            "synthetic_path": row.absolute_path,
            "synthetic_md5": synthetic_md5,
            "synthetic_dhash": synthetic_dhash,
            "exact_md5_match": synthetic_md5 in real_md5,
            "exact_md5_real_ids": ";".join(real_md5.get(synthetic_md5, [])),
            "min_dhash_hamming": int(hamming[nearest]),
            "perceptual_near_duplicate_hamming_le_2": int(hamming[nearest]) <= 2,
            "nearest_dhash_real_id": real.iloc[nearest]["image_id"],
            "nearest_dhash_real_split": real.iloc[nearest]["split"],
        })
        self_rows.append({"synthetic_id": row.image_id, "synthetic_md5": synthetic_md5, "synthetic_dhash": synthetic_dhash})
    pd.DataFrame(duplicate_rows).to_csv(output / "synthetic_real_duplicate_flags.csv", index=False)
    self_frame = pd.DataFrame(self_rows)
    self_frame["exact_md5_duplicate"] = self_frame.duplicated("synthetic_md5", keep=False)
    self_frame["exact_dhash_duplicate"] = self_frame.duplicated("synthetic_dhash", keep=False)
    self_frame.to_csv(output / "synthetic_synthetic_duplicate_flags.csv", index=False)

    sampled = []
    for _, block in unique.groupby("label", sort=True):
        sampled.append(block.sample(n=min(len(block), 100), random_state=seed))
    sampled = pd.concat(sampled, ignore_index=True) if sampled else unique.iloc[:0]
    rows = []
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        structural_similarity = None
    for row in sampled.itertuples(index=False):
        with Image.open(row.absolute_path).convert("L") as image:
            padded = base.letterbox_square(image.convert("RGB"), IMAGE_SIZE, pad_mode=pad_mode).convert("L")
            synthetic_array = np.asarray(padded, dtype=np.float32) / 255.0
            synthetic_embedding = embedding(padded)
        distances = np.linalg.norm(real_embeddings - synthetic_embedding[None, :], axis=1)
        nearest = int(np.argmin(distances))
        ssim = np.nan
        if structural_similarity is not None:
            ssim = float(structural_similarity(synthetic_array, real_arrays[nearest], data_range=1.0))
        rows.append({
            "seed": seed,
            "generation_method": method,
            "synthetic_id": row.image_id,
            "synthetic_label": row.label,
            "synthetic_path": row.absolute_path,
            "nearest_real_id": real.iloc[nearest]["image_id"],
            "nearest_real_path": real.iloc[nearest]["absolute_path"],
            "nearest_real_split": real.iloc[nearest]["split"],
            "ssim_224": ssim,
            "embedding_distance": float(distances[nearest]),
            "embedding_method": "letterbox224_grayscale32_gradient",
            "n_unique_synthetic": int(len(unique)),
            "n_sampled": int(len(sampled)),
        })
    similarity = pd.DataFrame(rows)
    similarity.to_csv(output / "synthetic_real_nearest_neighbors.csv", index=False)
    panel_rows = []
    for _, block in similarity.groupby("synthetic_label", sort=True):
        panel_rows.append(block.head(6))
    panel_frame = pd.concat(panel_rows, ignore_index=True) if panel_rows else similarity.iloc[:0]
    panels = []
    for row in panel_frame.itertuples(index=False):
        with Image.open(row.synthetic_path).convert("RGB") as image:
            left = image.resize((192, 192), Image.Resampling.LANCZOS)
        with Image.open(row.nearest_real_path).convert("RGB") as image:
            right = image.resize((192, 192), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (400, 230), "white")
        canvas.paste(left, (4, 4))
        canvas.paste(right, (204, 4))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 201), f"synthetic {row.synthetic_id}", fill="black")
        draw.text((206, 201), f"real {row.nearest_real_id}", fill="black")
        draw.text((6, 216), f"SSIM={row.ssim_224:.3f} dist={row.embedding_distance:.3f}", fill="black")
        panels.append(canvas)
    if panels:
        panel = Image.new("RGB", (400, 230 * len(panels)), "white")
        for index, image in enumerate(panels):
            panel.paste(image, (0, 230 * index))
        panel.save(output / "synthetic_real_nearest_neighbor_panels.png")
    (output / "similarity_complete.json").write_text(json.dumps({
        "status": "complete",
        "generation_method": method,
        "input_size": IMAGE_SIZE,
        "synthetic_md5_sha256": stable_token("\n".join(self_frame["synthetic_md5"].astype(str))),
        "n_unique_synthetic": int(len(unique)),
        "n_sampled": int(len(sampled)),
    }, indent=2), encoding="utf-8")


def save_split_audit(frame: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for column in ["md5", "dhash"]:
        for key, block in frame.groupby(column, dropna=False):
            splits = sorted(block["split"].astype(str).unique())
            if len(block) > 1:
                rows.append({"key_type": column, "key": key, "n_images": len(block), "splits": ";".join(splits), "cross_split": len(splits) > 1, "image_ids": ";".join(block["image_id"].astype(str))})
    frame.to_csv(output / "locked_split_audit_input.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "near_duplicate_audit.csv", index=False)
    split_counts = {
        f"{split}|{label_name}": int(count)
        for (split, label_name), count in frame.groupby(["split", "label_name"]).size().items()
    }
    (output / "split_audit.json").write_text(json.dumps({
        "group_field": PATIENT_GROUP_FIELD,
        "n_images": int(len(frame)),
        "n_patient_groups": int(frame["patient_group"].nunique()),
        "patient_group_crossings": int(frame.groupby("patient_group")["split"].nunique().gt(1).sum()),
        "exact_duplicate_link_crossings": int(base.count_cross_split_exact_links(frame)),
        "patient_group_definition": str(frame.attrs.get("patient_group_definition", "authoritative manifest patient_group")),
        "near_duplicate_links_crossing_splits": int(base.count_cross_split_near_duplicates(frame)),
        "mixed_label_groups": int(frame.groupby("patient_group")["label"].nunique().gt(1).sum()),
        "split_counts": split_counts,
    }, indent=2, default=str), encoding="utf-8")


def evaluate_route(
    spec: dict,
    frame: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    train_for_model: pd.DataFrame,
    synthetic: pd.DataFrame,
    route_dir: Path,
    seed: int,
    model,
    device,
    history: pd.DataFrame,
    configuration_fingerprint: str,
    stage: str = "full",
) -> None:
    head = spec["head"]
    quality = bool(spec["lidsentry"])
    train_x, train_y, train_ids = extract_features(model, train_for_model, quality, device)
    early_x, early_y, early_ids = extract_features(model, splits["early_stop"], quality, device)
    val_x, val_y, val_ids = extract_features(model, splits["val"], quality, device)
    feature_sets = {"train": train_x, "early_stop": early_x, "val": val_x}
    label_sets = {"train": train_y, "early_stop": early_y, "val": val_y}
    id_sets = {"train": train_ids, "early_stop": early_ids, "val": val_ids}
    if stage != "validation":
        for name in ["test1"] + (["source_holdout"] if not splits["source_holdout"].empty else []):
            values, labels, ids = extract_features(model, splits[name], quality, device)
            feature_sets[name], label_sets[name], id_sets[name] = values, labels, ids
    direct_probs = {}
    if head == "cnn":
        direct_names = ["val"] if stage == "validation" else ["val", "test1"] + (["source_holdout"] if not splits["source_holdout"].empty else [])
        for name in direct_names:
            direct_probs[name], _, _ = predict_model(model, splits[name], quality, device)
        probabilities = direct_probs
        head_model = None
    else:
        # The formal validation fold is reserved for route/threshold
        # selection.  XGBoost early stopping uses the independent training
        # early-stop split instead.
        head_model = fit_head(head, train_x, train_y, early_x, early_y, seed)
        probabilities = {
            name: _head_probabilities(head, head_model, values)
            for name, values in feature_sets.items()
            if name != "train"
        }
    selected = base.thresholds(label_sets["val"], probabilities["val"])
    (route_dir / "thresholds.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    metrics_rows, prediction_rows = [], []
    group_lookup = frame.set_index("image_id")["patient_group"]
    output_splits = ["val"] if stage == "validation" else ["val", "test1"] + (["source_holdout"] if not splits["source_holdout"].empty else [])
    for split_name in output_splits:
        y_true = label_sets[split_name]
        prob = probabilities[split_name]
        ids = id_sets[split_name]
        prediction_rows.append(pd.DataFrame({
            "route": route_name(spec),
            "seed": seed,
            "split": split_name,
            "image_id": ids,
            "label": y_true,
            "prob_malignant": prob,
            "patient_group": group_lookup.reindex(ids).fillna("unknown").to_numpy(),
            "generation_method": spec["generation_method"],
            "backbone": spec["backbone"],
            "classifier_head": head,
            "lidsentry": quality,
        }))
        for operating_point, threshold in selected.items():
            row = base.metrics(y_true, prob, float(threshold), route_name(spec), seed, split_name, float(spec["ratio"]), IMAGE_SIZE, quality, spec["strategy"])
            row.update({
                "generation_method": spec["generation_method"],
                "backbone": spec["backbone"],
                "classifier_head": head,
                "strong_conventional": bool(spec.get("strong_conventional")),
                "threshold_source": "validation_locked",
            })
            metrics_rows.append(row | {"operating_point": operating_point})
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_name = "validation_metrics.csv" if stage == "validation" else "metrics_operating_points.csv"
    predictions_name = "validation_predictions.csv" if stage == "validation" else "predictions.csv"
    metrics_frame.to_csv(route_dir / metrics_name, index=False)
    confusion_name = "validation_confusion_matrices.csv" if stage == "validation" else "confusion_matrices.csv"
    metrics_frame[[
        "route", "seed", "split", "operating_point", "threshold", "tp", "fp", "tn", "fn",
    ]].to_csv(route_dir / confusion_name, index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(route_dir / predictions_name, index=False)
    if head_model is not None:
        import joblib

        if head == "xgboost":
            head_model.save_model(str(route_dir / "xgboost_model.ubj"))
        else:
            joblib.dump(head_model, route_dir / f"{head}_model.joblib")
    torch = __import__("torch")
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "backbone": spec["backbone"],
        "configuration_fingerprint": configuration_fingerprint,
    }, route_dir / "backbone_best_state.pt")
    history.to_csv(route_dir / "history.csv", index=False)
    diagnostics = {
        "route": route_name(spec),
        "seed": seed,
        "paper": spec.get("paper"),
        "native_feature_dimension": int(getattr(model, "_native_feature_dim", 0)),
        "feature_dimension": int(getattr(model, "_projection_dim", PROJECTION_DIM)),
        "head_feature_dimension": int(
            getattr(model, "_projection_dim", PROJECTION_DIM)
            + (1 if quality and head != "cnn" else 0)
        ),
        "feature_projection": "native backbone feature -> trainable 512-D projection",
        "optimizer_updates_per_real_equivalent_epoch": int(history["optimizer_updates"].iloc[0]) if "optimizer_updates" in history.columns and not history.empty else 0,
        "best_epoch": int(history.iloc[0]["selected_best_epoch"]) if not history.empty else 0,
        "best_early_stop_auroc": float(history.iloc[0]["selected_best_early_stop_auroc"]) if not history.empty else np.nan,
        "completed_epochs": int(history["epoch"].max()) if not history.empty else 0,
        "restored_best_early_stop_state": True,
        "restored_best_validation_state": False,
        "early_stopping_monitor": "train_earlystop AUROC",
        "preprocessing": ("aspect_ratio_preserving_reflect_letterbox_224_imagenet_normalization"
                         if os.environ.get("MANUAL_EYE_DATASET_ROOT")
                         else "aspect_ratio_preserving_constant_letterbox_224_imagenet_normalization"),
        "xgboost_prediction_backend": getattr(
            head_model if head == "xgboost" else None,
            "_sycnn_prediction_backend",
            None,
        ),
    }
    (route_dir / "training_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
    if not synthetic.empty:
        save_similarity_audit(frame, synthetic, route_dir, seed, spec["generation_method"])


def load_saved_head(route_dir: Path, head: str):
    """Load a downstream head without refitting it on locked test images."""

    if head == "cnn":
        return None
    if head == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(str(route_dir / "xgboost_model.ubj"))
        return model
    import joblib

    return joblib.load(route_dir / f"{head}_model.joblib")


def evaluate_saved_route(
    spec: dict,
    frame: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    route_dir: Path,
    seed: int,
) -> None:
    """Run locked test evaluation from immutable validation artifacts.

    A completed test route may be re-entered after an AutoDL restart; the
    route manifest is then marked ``test`` but its validation predictions and
    fitted head remain the same.  Re-running the locked evaluation is safe and
    never refits on test images.
    """

    manifest_path = route_dir / "run_manifest.json"
    validation_predictions_path = route_dir / "validation_predictions.csv"
    thresholds_path = route_dir / "thresholds.json"
    if not manifest_path.is_file() or not validation_predictions_path.is_file() or not thresholds_path.is_file():
        raise RuntimeError(f"Cannot run test stage before validation artifacts exist: {route_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") not in {"validation", "test"}:
        raise RuntimeError(f"Route was not produced by the validation/test protocol: {route_dir}")
    model, _, device = load_saved_representation(spec, route_dir)
    quality = bool(manifest.get("lidsentry", False))
    head = str(manifest.get("head", spec["head"]))
    head_model = load_saved_head(route_dir, head)
    group_lookup = frame.set_index("image_id")["patient_group"]
    predictions = [pd.read_csv(validation_predictions_path)]
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    evaluation_frames = {"test1": splits["test1"]}
    if not splits["source_holdout"].empty:
        evaluation_frames["source_holdout"] = splits["source_holdout"]
    for split_name, split_frame in evaluation_frames.items():
        if head == "cnn":
            probabilities, labels, ids = predict_model(model, split_frame, quality, device)
        else:
            features, labels, ids = extract_features(model, split_frame, quality, device)
            probabilities = _head_probabilities(head, head_model, features)
        predictions.append(pd.DataFrame({
            "route": route_name(spec),
            "seed": seed,
            "split": split_name,
            "image_id": ids,
            "label": labels,
            "prob_malignant": probabilities,
            "patient_group": group_lookup.reindex(ids).fillna("unknown").to_numpy(),
            "generation_method": manifest.get("generation_method"),
            "backbone": manifest.get("backbone"),
            "classifier_head": head,
            "lidsentry": quality,
        }))
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metrics_rows = []
    for split_name in ["val", "test1", "source_holdout"]:
        current = prediction_frame[prediction_frame["split"].astype(str).eq(split_name)]
        if current.empty:
            continue
        for operating_point, threshold in thresholds.items():
            row = base.metrics(
                current["label"].to_numpy(), current["prob_malignant"].to_numpy(),
                float(threshold), route_name(spec), seed, split_name,
                float(manifest.get("ratio", 0)), IMAGE_SIZE, quality,
                str(manifest.get("strategy", "unknown")),
            )
            row.update({
                "generation_method": manifest.get("generation_method"),
                "backbone": manifest.get("backbone"),
                "classifier_head": head,
                "strong_conventional": bool(manifest.get("strong_conventional")),
                "threshold_source": "validation_locked",
                "operating_point": operating_point,
            })
            metrics_rows.append(row)
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(route_dir / "metrics_operating_points.csv", index=False)
    metrics_frame[[
        "route", "seed", "split", "operating_point", "threshold", "tp", "fp", "tn", "fn",
    ]].to_csv(route_dir / "confusion_matrices.csv", index=False)
    prediction_frame.to_csv(route_dir / "predictions.csv", index=False)
    manifest.update({
        "stage": "test",
        "test_evaluation_completed": True,
        "test_metrics_used_for_selection": False,
        "test_evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (route_dir / "run_complete.json").write_text(json.dumps({
        "route": route_name(spec), "seed": seed, "status": "complete",
        "configuration_fingerprint": manifest.get("configuration_fingerprint"),
        "stage": "test",
    }, indent=2), encoding="utf-8")


def run_route(
    spec: dict,
    frame: pd.DataFrame,
    split_audit: dict,
    splits: dict[str, pd.DataFrame],
    seed: int,
    pools: dict[str, Path],
    representation_cache: dict[str, tuple] | None = None,
    stage: str = "full",
) -> Path:
    route = route_name(spec)
    route_dir = base.OUT / "full_highres_matrix" / f"seed_{seed}" / route
    route_dir.mkdir(parents=True, exist_ok=True)
    if stage == "test":
        evaluate_saved_route(spec, frame, splits, route_dir, seed)
        return route_dir
    train = splits["train"].copy()
    pool = pools.get(spec["generation_method"])
    synthetic = make_synthetic_frame(pool, train, float(spec["ratio"]), spec["strategy"], seed, spec["generation_method"])
    train_for_model = pd.concat([train, synthetic], ignore_index=True)
    split_source = Path(os.environ.get("LOCKED_SPLIT", str(BUNDLE / "locked_split.csv")))
    native_feature_dimension = int(NATIVE_FEATURE_DIMS[spec["backbone"]])
    generation_configuration_fingerprint = None
    synthetic_pool_image_count = 0
    if float(spec["ratio"]) > 0:
        generation_marker = pool / f"{spec['generation_method']}_complete.json"
        if not generation_marker.is_file():
            raise RuntimeError(f"Missing completed {spec['generation_method']} marker: {generation_marker}")
        generation_payload = json.loads(generation_marker.read_text(encoding="utf-8"))
        generation_configuration_fingerprint = generation_payload.get("configuration_fingerprint")
        if not generation_configuration_fingerprint:
            raise RuntimeError(f"Generation marker has no configuration fingerprint: {generation_marker}")
        synthetic_pool_image_count = len(list(pool.glob("pool_class*.png")))
    representation_dir = representation_artifact_dir(spec, seed)
    route_config = {
        **spec,
        "route": route,
        "seed": seed,
        "paper": spec.get("paper"),
        "group_field": PATIENT_GROUP_FIELD,
        "input_size": IMAGE_SIZE,
        "generator_size": GENERATOR_SIZE,
        "preprocessing": "aspect_ratio_preserving_letterbox_224_imagenet_normalization",
        "backbone_initialization": "imagenet",
        "pretrained_weights_used": True,
        "native_feature_dimension": native_feature_dimension,
        "feature_dimension": PROJECTION_DIM,
        "head_feature_dimension": PROJECTION_DIM + (1 if spec["lidsentry"] and spec["head"] != "cnn" else 0),
        "feature_projection": f"{native_feature_dimension}-D native feature -> trainable {PROJECTION_DIM}-D projection",
        "representation_key": representation_key(spec),
        "representation_artifact_dir": representation_artifact_relpath(spec),
        "requested_synthetic_ratio": float(spec["ratio"]),
        "realized_synthetic_ratio": float(len(synthetic) / max(len(train), 1)),
        "real_training_images": int(len(train)),
        "early_stop_images": int(len(splits["early_stop"])),
        "formal_validation_images": int(len(splits["val"])),
        "formal_test_images": int(len(splits["test1"])),
        "synthetic_training_images": int(len(synthetic)),
        "final_training_images": int(len(train_for_model)),
        "optimizer_updates_per_real_equivalent_epoch": int(math.ceil(len(train) / base.BATCH_SIZE)),
        "final_class_counts": {str(key): int(value) for key, value in train_for_model["label"].value_counts().to_dict().items()},
        "manifest_sha256": base.sha256_file(Path(os.environ["MANIFEST_PATH"])),
        "generation_configuration_fingerprint": generation_configuration_fingerprint,
        "synthetic_pool_image_count": synthetic_pool_image_count,
        "training_image_ids_sha256": hashlib.sha256(
            "\n".join(sorted(train_for_model["image_id"].astype(str))).encode("utf-8")
        ).hexdigest(),
        "lidsentry_components": ["quality_conditioned_training_degradation", "quality_weighted_backbone_loss", "quality_scalar_feature"] if spec["lidsentry"] else [],
        "locked_split_sha256": base.sha256_file(split_source),
        "code_sha256": base.sha256_file(Path(__file__).resolve()),
        "base_runner_sha256": base.sha256_file(BUNDLE / "reproduce_gan_resnet50_xgb_patient224.py"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "package_versions": base.package_versions(),
    }
    representation_config = {
        key: route_config[key]
        for key in [
            "paper", "generation_method", "backbone", "ratio", "strategy", "lidsentry",
            "strong_conventional", "input_size", "generator_size", "backbone_initialization",
            "pretrained_weights_used", "native_feature_dimension", "feature_dimension",
            "feature_projection", "optimizer_updates_per_real_equivalent_epoch",
            "manifest_sha256", "locked_split_sha256", "training_image_ids_sha256",
        ]
        if key in route_config
    }
    route_config["representation_fingerprint"] = hashlib.sha256(
        json.dumps(representation_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    representation_complete = representation_dir / "representation_complete.json"
    configuration_fingerprint = hashlib.sha256(json.dumps(route_config, sort_keys=True).encode("utf-8")).hexdigest()
    complete = route_dir / "run_complete.json"
    validation_complete = route_dir / "validation_complete.json"
    completion_marker = validation_complete if stage == "validation" else complete
    required = [
        route_dir / "training_manifest.csv",
        route_dir / "early_stop_manifest.csv",
        route_dir / ("validation_metrics.csv" if stage == "validation" else "metrics_operating_points.csv"),
        route_dir / ("validation_confusion_matrices.csv" if stage == "validation" else "confusion_matrices.csv"),
        route_dir / ("validation_predictions.csv" if stage == "validation" else "predictions.csv"),
        route_dir / "thresholds.json",
        route_dir / "history.csv",
        route_dir / "training_diagnostics.json",
        route_dir / "backbone_best_state.pt",
        route_dir / "run_manifest.json",
        completion_marker,
        representation_dir / "backbone_best_state.pt",
        representation_dir / "history.csv",
        representation_complete,
    ]
    if spec["head"] == "xgboost":
        required.append(route_dir / "xgboost_model.ubj")
    elif spec["head"] != "cnn":
        required.append(route_dir / f"{spec['head']}_model.joblib")
    if float(spec["ratio"]) > 0:
        required.extend([
            route_dir / "similarity" / "synthetic_real_nearest_neighbors.csv",
            route_dir / "similarity" / "synthetic_real_duplicate_flags.csv",
            route_dir / "similarity" / "synthetic_synthetic_duplicate_flags.csv",
            route_dir / "similarity" / "synthetic_real_nearest_neighbor_panels.png",
        ])
    cache_key = representation_key(spec)
    cached_representation = representation_cache.get(cache_key) if representation_cache is not None else None
    marker = validation_complete if stage == "validation" else complete
    expected_status = "validation_complete" if stage == "validation" else "complete"
    route_is_complete = marker.is_file() and not base.FORCE_RERUN and all(path.is_file() and path.stat().st_size > 0 for path in required)
    if route_is_complete:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("configuration_fingerprint") == configuration_fingerprint and payload.get("status") == expected_status:
                if cached_representation is None and representation_cache is not None:
                    cached_representation = load_saved_representation(spec, route_dir)
                    representation_cache[cache_key] = cached_representation
                print(f"[SKIP] full matrix route exists: {route}")
                return route_dir
        except Exception:
            pass
    training_manifest = train_for_model.copy()
    training_manifest["generation_source"] = np.where(training_manifest["synthetic"].astype(bool), f"{spec['generation_method']}_train_only", "real_train_only")
    training_manifest["requested_synthetic_ratio"] = float(spec["ratio"])
    training_manifest["realized_synthetic_ratio"] = float(len(synthetic) / max(len(train), 1))
    training_manifest["class_generation_strategy"] = spec["strategy"]
    training_manifest["generation_method"] = spec["generation_method"]
    training_manifest.to_csv(route_dir / "training_manifest.csv", index=False)
    splits["early_stop"].to_csv(route_dir / "early_stop_manifest.csv", index=False)
    if cached_representation is None:
        shared_is_complete = False
        if representation_complete.is_file() and not base.FORCE_RERUN:
            try:
                shared_payload = json.loads(representation_complete.read_text(encoding="utf-8"))
                shared_is_complete = (
                    shared_payload.get("status") == "complete"
                    and shared_payload.get("representation_fingerprint") == route_config["representation_fingerprint"]
                )
            except (OSError, json.JSONDecodeError):
                shared_is_complete = False
        if shared_is_complete:
            cached_representation = load_representation_from_dir(spec, representation_dir)
        else:
            cached_representation = train_model(
                train_for_model,
                splits["early_stop"],
                spec["backbone"],
                bool(spec["lidsentry"]),
                bool(spec.get("strong_conventional")),
                seed,
                representation_dir,
                route_config["representation_fingerprint"],
                real_training_size=len(train),
            )
            save_shared_representation(
                spec,
                seed,
                representation_dir,
                cached_representation[0],
                cached_representation[1],
                route_config["representation_fingerprint"],
            )
        if representation_cache is not None:
            representation_cache[cache_key] = cached_representation
    model, history, device = cached_representation
    evaluate_route(
        spec, frame, splits, train_for_model, synthetic, route_dir, seed,
        model, device, history, configuration_fingerprint, stage=stage,
    )
    # Keep the route-local compatibility artifact byte-identical to the
    # shared representation so interrupted multi-head jobs cannot silently
    # compare heads fitted from different neural states.
    shutil.copyfile(
        representation_dir / "backbone_best_state.pt",
        route_dir / "backbone_best_state.pt",
    )
    shutil.copyfile(representation_dir / "history.csv", route_dir / "history.csv")
    route_config.update({
        "device_for_training": str(device),
        "cuda_device_name": __import__("torch").cuda.get_device_name(device) if str(device).startswith("cuda") else None,
        "split_audit": split_audit,
        "representation_checkpoint_sha256": base.sha256_file(
            representation_dir / "backbone_best_state.pt"
        ),
        "representation_history_sha256": base.sha256_file(representation_dir / "history.csv"),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if stage != "validation" else "validation_complete",
        "stage": stage,
        "configuration_fingerprint": configuration_fingerprint,
    })
    (route_dir / "run_manifest.json").write_text(json.dumps(route_config, indent=2, default=str), encoding="utf-8")
    if stage == "validation":
        validation_complete.write_text(json.dumps({
            "route": route, "seed": seed, "status": "validation_complete",
            "configuration_fingerprint": configuration_fingerprint, "stage": "validation",
        }, indent=2), encoding="utf-8")
    else:
        complete.write_text(json.dumps({
            "route": route, "seed": seed, "status": "complete",
            "configuration_fingerprint": configuration_fingerprint, "stage": stage,
        }, indent=2), encoding="utf-8")
    checkpoint = route_dir / "backbone_checkpoint.pt"
    if checkpoint.exists():
        checkpoint.unlink()
    return route_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=os.environ.get("FULL_MATRIX_COHORT", "retrospective_cohort"), choices=["retrospective_cohort"])
    parser.add_argument("--methods")
    parser.add_argument("--backbones")
    parser.add_argument("--heads")
    parser.add_argument("--ratios")
    parser.add_argument("--quality")
    parser.add_argument("--strategies")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--include-strong", action="store_true", default=None)
    parser.add_argument("--stage", choices=["validation", "test", "full"], default=os.environ.get("FULL_MATRIX_STAGE", "validation"))
    parser.add_argument("--selection-lock", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-routes", type=int, default=int(os.environ.get("FULL_MATRIX_MAX_ROUTES", "0")))
    args = parser.parse_args()
    defaults = cohort_defaults(args.cohort)
    methods = parse_csv(args.methods or os.environ.get("FULL_MATRIX_METHODS"), tuple(defaults["methods"]))
    backbones = parse_csv(args.backbones or os.environ.get("FULL_MATRIX_BACKBONES"), tuple(defaults["backbones"]))
    heads = parse_csv(args.heads or os.environ.get("FULL_MATRIX_HEADS"), tuple(defaults["heads"]))
    ratios = parse_ratios(args.ratios or os.environ.get("FULL_MATRIX_RATIOS")) if args.ratios or os.environ.get("FULL_MATRIX_RATIOS") else list(defaults["ratios"])
    quality_values = parse_quality_tokens(
        args.quality or os.environ.get("FULL_MATRIX_QUALITY"),
        tuple(defaults["quality"]),
    )
    strategies = parse_csv(args.strategies or os.environ.get("FULL_MATRIX_STRATEGIES"), tuple(defaults["strategies"]))
    include_strong = args.include_strong if args.include_strong is not None else parse_bool(
        os.environ.get("FULL_MATRIX_INCLUDE_STRONG"), bool(defaults["include_strong"])
    )
    validate_cohort_protocol(
        args.cohort, methods, backbones, heads, ratios, quality_values, strategies, include_strong
    )
    if args.max_routes > 0 and not args.dry_run:
        raise ValueError("--max-routes is a dry-run inspection option; formal runs must execute the complete locked matrix")
    specs = build_specs(methods, backbones, heads, ratios, quality_values, strategies, include_strong)
    for spec in specs:
        spec["paper"] = args.cohort
    if args.stage == "test":
        if args.selection_lock is None or not args.selection_lock.is_file():
            raise ValueError("--selection-lock is required for the test stage")
        lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
        selected_routes = set(lock.get("selected_routes", []))
        if not selected_routes:
            raise ValueError("selection lock does not contain selected_routes")
        specs = [spec for spec in specs if route_name(spec) in selected_routes]
        if not specs:
            raise ValueError("selection lock does not match any routes in the selected paper protocol")
    if args.max_routes > 0:
        specs = specs[: args.max_routes]
    write_shared_artifacts = (
        not os.environ.get("MANUAL_EYE_DATASET_ROOT")
        or args.seed == int(os.environ.get("RETROSPECTIVE_SHARED_PLAN_SEED", "17"))
    )
    print(json.dumps({
        "seed": args.seed,
        "paper": args.cohort,
        "methods": methods,
        "backbones": backbones,
        "heads": heads,
        "ratios": ratios,
        "lidsentry": quality_values,
        "strategies": strategies,
        "include_strong": include_strong,
        "route_count": len(specs),
        "image_size": IMAGE_SIZE,
        "generator_size": GENERATOR_SIZE,
        "feature_dimension": PROJECTION_DIM,
    }, indent=2))
    # Local manual-cohort dry runs validate the route contract without writing
    # a plan into the server's default output location.
    if args.dry_run and os.environ.get("MANUAL_EYE_DATASET_ROOT"):
        return 0
    if write_shared_artifacts:
        base.OUT.mkdir(parents=True, exist_ok=True)
        plan_frame = pd.DataFrame([{**spec, "route": route_name(spec)} for spec in specs])
        plan_name = (
            "validation_route_plan.csv"
            if args.stage == "validation"
            else "locked_test_route_plan.csv"
        )
        plan_frame.to_csv(base.OUT / plan_name, index=False)
        # Keep the complete validation plan as the canonical matrix plan.  The
        # locked test stage must never overwrite it with the smaller selected set.
        if args.stage == "validation":
            plan_frame.to_csv(base.OUT / "full_highres_matrix_route_plan.csv", index=False)
    if args.dry_run:
        return 0
    frame, split_audit, splits = load_locked_data()
    matrix_root = base.OUT / "full_highres_matrix"
    matrix_root.mkdir(parents=True, exist_ok=True)
    if write_shared_artifacts:
        save_split_audit(frame, matrix_root)
        splits["locked_train"].to_csv(matrix_root / "train_earlystop_manifest.csv", index=False)
        (matrix_root / "train_earlystop_split.json").write_text(json.dumps({
            "status": "locked",
            "assignment_seed": 1729,
            "group_field": PATIENT_GROUP_FIELD,
            "train_fit_images": int(len(splits["train"])),
            "train_earlystop_images": int(len(splits["early_stop"])),
            "train_fit_groups": int(splits["train"][PATIENT_GROUP_FIELD].nunique()),
            "train_earlystop_groups": int(splits["early_stop"][PATIENT_GROUP_FIELD].nunique()),
            "validation_not_used_for_early_stopping": True,
        }, indent=2), encoding="utf-8")
    needed_methods = {spec["generation_method"] for spec in specs if float(spec["ratio"]) > 0} if args.stage != "test" else set()
    pools: dict[str, Path] = {}
    for method in sorted(needed_methods):
        if method == "gan":
            pools[method] = base.train_gan(splits["train"], args.seed)
        elif method == "diffusion":
            pools[method] = train_diffusion(splits["train"], args.seed)
        else:
            raise ValueError(f"Unsupported generation method: {method}")
        save_generation_quality_audit(splits["train"], pools[method], method, args.seed)
    route_index = []
    representation_cache: dict[str, tuple] = {}
    for index, spec in enumerate(specs, start=1):
        print(f"[MATRIX] {index}/{len(specs)} {route_name(spec)}")
        route_dir = run_route(spec, frame, split_audit, splits, args.seed, pools, representation_cache, stage=args.stage)
        route_index.append({**spec, "route": route_name(spec), "seed": args.seed, "route_dir": str(route_dir), "status": "complete"})
    pd.DataFrame(route_index).to_csv(matrix_root / f"route_index_seed_{args.seed}.csv", index=False)
    (matrix_root / f"matrix_complete_seed_{args.seed}.json").write_text(json.dumps({
        "status": "complete",
        "stage": args.stage,
        "seed": args.seed,
        "route_count": len(specs),
        "routes": [route_name(spec) for spec in specs],
        "split_audit": split_audit,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
