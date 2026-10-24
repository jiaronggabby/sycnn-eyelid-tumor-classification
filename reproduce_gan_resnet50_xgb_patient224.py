from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parent
SEEDS = [17, 42, 2024, 2025, 3407]
RATIOS = [0, 0.5, 1, 2, 3, 4, 5]
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]
LABEL_MAP = {"Positive": 1, "Negative": 0}
# Kaggle batch kernels are uploaded as /kaggle/src/script.py, so local folder
# names are not reliable runtime configuration. Each submitted copy embeds this.
REMOTE_RUNTIME = None


def runtime_config() -> dict:
    path = ROOT / "runtime.json"
    payload = dict(REMOTE_RUNTIME or {})
    if not payload and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload:
        name = ROOT.name.lower()
        match = re.search(r"seed(17|42|2024|2025|3407)", name)
        if match:
            payload["seed"] = int(match.group(1))
        if "legacy" in name:
            payload["stage"] = "legacy"
        elif "ablation" in name:
            payload["stage"] = "primary_ablation"
        elif "robustness" in name:
            payload["stage"] = "robustness"
        elif "primary" in name:
            payload["stage"] = "primary_sweep"
        else:
            payload["stage"] = "audit"
        if "low" in name:
            payload["ratios"] = [0, 0.5, 1, 2]
        elif "high" in name:
            payload["ratios"] = [3, 4, 5]
        if payload.get("stage") == "primary_sweep":
            payload["include_strong_baseline"] = "strong" in name
            payload["include_source_holdout"] = True
    payload["stage"] = os.environ.get("RUN_STAGE", payload.get("stage", "audit"))
    payload["seed"] = int(os.environ.get("RUN_SEED", payload.get("seed", SEEDS[0])))
    ratios = os.environ.get("RUN_RATIOS")
    if ratios:
        payload["ratios"] = [float(value.strip()) for value in ratios.split(",") if value.strip()]
    if "INCLUDE_STRONG_BASELINE" in os.environ:
        payload["include_strong_baseline"] = os.environ["INCLUDE_STRONG_BASELINE"].strip().lower() in {"1", "true", "yes"}
    if "INCLUDE_SOURCE_HOLDOUT" in os.environ:
        payload["include_source_holdout"] = os.environ["INCLUDE_SOURCE_HOLDOUT"].strip().lower() in {"1", "true", "yes"}
    return payload


CFG = runtime_config()
OUT = Path(os.environ.get("OUTPUT_ROOT", "/kaggle/working/reproduction_outputs"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/kaggle/working/data/eyelid"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", CFG.get("image_size", 224)))
# The clinical classifier is evaluated at 224 px. The formal generator size is
# 256 px; a 512 px size is available only to explicitly marked quality pilots.
GAN_SIZE = int(os.environ.get("GAN_SIZE", 256))
ALLOW_EXPERIMENTAL_GENERATOR_SIZE = os.environ.get(
    "ALLOW_EXPERIMENTAL_GENERATOR_SIZE", "0"
).strip().lower() in {"1", "true", "yes"}
GAN_BATCH_SIZE = int(os.environ.get("GAN_BATCH_SIZE", 256))
GAN_GENERATOR_CHANNEL_PROFILE = os.environ.get(
    "GAN_GENERATOR_CHANNEL_PROFILE", "default"
).strip().lower()
GAN_GENERATOR_CHANNEL_PROFILES = {
    "default": [512, 256, 128, 64, 32, 16, 8],
    # Keep the first block unchanged and retain more channels at the
    # high-resolution stages, where the eye boundary and iris highlights are
    # formed. This is an explicit capacity ablation, not a change to the loss.
    "wide_late": [512, 384, 256, 128, 64, 32, 16],
}
if GAN_GENERATOR_CHANNEL_PROFILE not in GAN_GENERATOR_CHANNEL_PROFILES:
    raise ValueError(
        "Unsupported GAN_GENERATOR_CHANNEL_PROFILE="
        f"{GAN_GENERATOR_CHANNEL_PROFILE!r}; expected one of "
        f"{sorted(GAN_GENERATOR_CHANNEL_PROFILES)}"
    )
BACKBONE_INIT = os.environ.get("BACKBONE_INIT", "imagenet").strip().lower()
EPOCHS = int(os.environ.get("EPOCHS", 300))
GAN_EPOCHS = int(os.environ.get("GAN_EPOCHS", 1000))
PATIENCE = int(os.environ.get("PATIENCE", 30))
CNN_MIN_EPOCHS = int(os.environ.get("CNN_MIN_EPOCHS", 60))
CNN_MIN_DELTA = float(os.environ.get("CNN_MIN_DELTA", 1e-4))
GAN_PATIENCE = int(os.environ.get("GAN_PATIENCE", 10))
GAN_MIN_EPOCHS = int(os.environ.get("GAN_MIN_EPOCHS", 1))
GAN_PATIENCE_INTERVAL_EPOCHS = int(os.environ.get("GAN_PATIENCE_INTERVAL_EPOCHS", 1))
if GAN_PATIENCE < 1 or GAN_MIN_EPOCHS < 1 or GAN_PATIENCE_INTERVAL_EPOCHS < 1:
    raise ValueError("GAN_PATIENCE, GAN_MIN_EPOCHS, and GAN_PATIENCE_INTERVAL_EPOCHS must be positive")
GAN_MIN_DELTA = float(os.environ.get("GAN_MIN_DELTA", 1e-3))
GAN_STRUCTURE_FIRST = os.environ.get("GAN_STRUCTURE_FIRST", "1").strip().lower() in {"1", "true", "yes"}
GAN_OBJECTIVE = os.environ.get(
    "GAN_OBJECTIVE", "hinge" if GAN_STRUCTURE_FIRST else "bce"
).strip().lower()
if GAN_OBJECTIVE not in {"bce", "hinge"}:
    raise ValueError(f"Unsupported GAN_OBJECTIVE={GAN_OBJECTIVE!r}; expected bce or hinge")
GAN_SPECTRAL_NORM = os.environ.get(
    "GAN_SPECTRAL_NORM", "1" if GAN_STRUCTURE_FIRST else "0"
).strip().lower() in {"1", "true", "yes"}
GAN_DISCRIMINATOR_PATCH = os.environ.get("GAN_DISCRIMINATOR_PATCH", "0").strip().lower() in {"1", "true", "yes"}
GAN_DISCRIMINATOR_MULTISCALE = os.environ.get(
    "GAN_DISCRIMINATOR_MULTISCALE", "1" if GAN_STRUCTURE_FIRST else "0"
).strip().lower() in {"1", "true", "yes"}
GAN_DISCRIMINATOR_PATCH_BLOCKS = int(
    os.environ.get("GAN_DISCRIMINATOR_PATCH_BLOCKS", "3" if GAN_STRUCTURE_FIRST else "0")
)
if GAN_DISCRIMINATOR_PATCH_BLOCKS < 0:
    raise ValueError("GAN_DISCRIMINATOR_PATCH_BLOCKS must be non-negative")
if GAN_DISCRIMINATOR_PATCH and GAN_DISCRIMINATOR_MULTISCALE:
    raise ValueError("Choose either a patch-only or multiscale discriminator")
GAN_SHARPNESS_SELECTION_WEIGHT = float(os.environ.get("GAN_SHARPNESS_SELECTION_WEIGHT", "0.0"))
if not math.isfinite(GAN_SHARPNESS_SELECTION_WEIGHT) or GAN_SHARPNESS_SELECTION_WEIGHT < 0:
    raise ValueError("GAN_SHARPNESS_SELECTION_WEIGHT must be a finite, non-negative number")
GAN_LAPLACIAN_MATCH_WEIGHT = float(os.environ.get("GAN_LAPLACIAN_MATCH_WEIGHT", "0.0"))
if not math.isfinite(GAN_LAPLACIAN_MATCH_WEIGHT) or GAN_LAPLACIAN_MATCH_WEIGHT < 0:
    raise ValueError("GAN_LAPLACIAN_MATCH_WEIGHT must be a finite, non-negative number")
GAN_FEATURE_MATCH_WEIGHT = float(
    os.environ.get("GAN_FEATURE_MATCH_WEIGHT", "3.0" if GAN_STRUCTURE_FIRST else "0.0")
)
if not math.isfinite(GAN_FEATURE_MATCH_WEIGHT) or GAN_FEATURE_MATCH_WEIGHT < 0:
    raise ValueError("GAN_FEATURE_MATCH_WEIGHT must be a finite, non-negative number")
GAN_FEATURE_MATCH_VARIANCE_WEIGHT = float(
    os.environ.get("GAN_FEATURE_MATCH_VARIANCE_WEIGHT", "0.0")
)
if not math.isfinite(GAN_FEATURE_MATCH_VARIANCE_WEIGHT) or GAN_FEATURE_MATCH_VARIANCE_WEIGHT < 0:
    raise ValueError("GAN_FEATURE_MATCH_VARIANCE_WEIGHT must be a finite, non-negative number")
GAN_EDGE_STRUCTURE_MATCH_WEIGHT = float(
    os.environ.get("GAN_EDGE_STRUCTURE_MATCH_WEIGHT", "0.0")
)
if not math.isfinite(GAN_EDGE_STRUCTURE_MATCH_WEIGHT) or GAN_EDGE_STRUCTURE_MATCH_WEIGHT < 0:
    raise ValueError("GAN_EDGE_STRUCTURE_MATCH_WEIGHT must be a finite, non-negative number")
GAN_EYE_STRUCTURE_LOSS_WEIGHT = float(os.environ.get("GAN_EYE_STRUCTURE_LOSS_WEIGHT", "0.0"))
GAN_EYE_STRUCTURE_SELECTION_WEIGHT = float(os.environ.get("GAN_EYE_STRUCTURE_SELECTION_WEIGHT", "0.0"))
GAN_STRUCTURE_CONDITION = os.environ.get(
    "GAN_STRUCTURE_CONDITION", "0"
).strip().lower() in {"1", "true", "yes"}
if not math.isfinite(GAN_EYE_STRUCTURE_LOSS_WEIGHT) or GAN_EYE_STRUCTURE_LOSS_WEIGHT < 0:
    raise ValueError("GAN_EYE_STRUCTURE_LOSS_WEIGHT must be a finite, non-negative number")
if not math.isfinite(GAN_EYE_STRUCTURE_SELECTION_WEIGHT) or GAN_EYE_STRUCTURE_SELECTION_WEIGHT < 0:
    raise ValueError("GAN_EYE_STRUCTURE_SELECTION_WEIGHT must be a finite, non-negative number")
GAN_DISCRIMINATOR_AUGMENT = os.environ.get(
    "GAN_DISCRIMINATOR_AUGMENT", "1" if GAN_STRUCTURE_FIRST else "0"
).strip().lower() in {"1", "true", "yes"}
GAN_DISCRIMINATOR_AUGMENT_PROB = float(os.environ.get("GAN_DISCRIMINATOR_AUGMENT_PROB", "0.5"))
GAN_DISCRIMINATOR_AUGMENT_TRANSLATION = int(
    os.environ.get("GAN_DISCRIMINATOR_AUGMENT_TRANSLATION", "4")
)
GAN_HIGH_PASS_DISCRIMINATOR = os.environ.get(
    "GAN_HIGH_PASS_DISCRIMINATOR", "0"
).strip().lower() in {"1", "true", "yes"}
GENERATOR_PAD_MODE = os.environ.get("GENERATOR_PAD_MODE", "reflect").strip().lower()
if GENERATOR_PAD_MODE not in {"reflect", "constant", "median", "center_crop_extreme"}:
    raise ValueError(
        "GENERATOR_PAD_MODE must be 'reflect', 'constant', 'median', or 'center_crop_extreme'"
    )
GENERATOR_EXTREME_ASPECT_RATIO = float(os.environ.get("GENERATOR_EXTREME_ASPECT_RATIO", "1.75"))
if not math.isfinite(GENERATOR_EXTREME_ASPECT_RATIO) or GENERATOR_EXTREME_ASPECT_RATIO <= 1:
    raise ValueError("GENERATOR_EXTREME_ASPECT_RATIO must be finite and greater than 1")
if not 0 <= GAN_DISCRIMINATOR_AUGMENT_PROB <= 1:
    raise ValueError("GAN_DISCRIMINATOR_AUGMENT_PROB must be between 0 and 1")
if GAN_DISCRIMINATOR_AUGMENT_TRANSLATION < 0:
    raise ValueError("GAN_DISCRIMINATOR_AUGMENT_TRANSLATION must be non-negative")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
N_BOOT = int(os.environ.get("N_BOOT", 1000))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4 if os.name != "nt" else 0))
FORCE_RERUN = os.environ.get("FORCE_RERUN", "0").strip().lower() in {"1", "true", "yes"}
SKIP_POSTPROCESS = os.environ.get("SKIP_POSTPROCESS", "0").strip().lower() in {"1", "true", "yes"}
REQUIRE_CUDA = os.environ.get("REQUIRE_CUDA", "0").strip().lower() in {"1", "true", "yes"}
VALID_STAGES = {
    "audit", "generate", "legacy", "primary", "primary_sweep", "primary_ablation",
    "balanced_ablation", "strong_baseline", "robustness", "analyze", "full", "full_matrix",
}
if str(CFG.get("stage", "audit")).lower() not in VALID_STAGES:
    raise ValueError(f"Unknown RUN_STAGE={CFG.get('stage')!r}; expected one of {sorted(VALID_STAGES)}")
if IMAGE_SIZE != 224:
    raise ValueError(f"This locked revision requires IMAGE_SIZE=224, got {IMAGE_SIZE}")
allowed_gan_sizes = {256, 512}
if ALLOW_EXPERIMENTAL_GENERATOR_SIZE:
    # The 128 px pilot is a controlled resolution bridge to the native
    # resolution of the historical GAN examples; it is not a formal
    # classifier-generation route.
    allowed_gan_sizes.add(128)
if GAN_SIZE not in allowed_gan_sizes or (GAN_SIZE in {128, 512} and not ALLOW_EXPERIMENTAL_GENERATOR_SIZE):
    raise ValueError(
        f"Formal runs require GAN_SIZE=256; only an explicit quality pilot may use 128 or 512, got {GAN_SIZE}"
    )
if BACKBONE_INIT != "imagenet":
    raise ValueError("This primary revision requires BACKBONE_INIT=imagenet")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def capture_rng_state() -> dict:
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].detach().to(device="cpu", dtype=torch.uint8))
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([value.detach().to(device="cpu", dtype=torch.uint8)
                                      for value in state["torch_cuda"]])


def device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if REQUIRE_CUDA:
            raise RuntimeError("CUDA is required for this formal run but torch.cuda.is_available() is false")
        return "cpu"
    except Exception:
        if REQUIRE_CUDA:
            raise
        return "cpu"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    versions = {}
    for name in ["numpy", "pandas", "Pillow", "scikit-learn", "scikit-image", "xgboost", "joblib", "torch", "torchvision", "openpyxl"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path, size: int = 8) -> str:
    with Image.open(path) as image:
        gray = ImageOps.grayscale(image).resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return f"{int(''.join('1' if bit else '0' for bit in bits.ravel()), 2):0{size * size // 4}x}"


def find_zip() -> Path:
    candidates = [
        Path("/kaggle/input/qarel-eyelid-images/images.zip"),
        Path("/kaggle/input/eyelid-tumor-risk-data/images.zip"),
    ]
    explicit = os.environ.get("IMAGE_ZIP")
    if explicit:
        candidates.insert(0, Path(explicit))
    candidates.extend(Path("/kaggle/input").rglob("images.zip"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    download_dir = Path("/tmp/repro_eyelid_download")
    download_dir.mkdir(parents=True, exist_ok=True)
    dataset = os.environ.get("DATASET_SLUG", "vsdavsdvmkk/qarel-eyelid-images")
    if not any(download_dir.glob("*.zip")):
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", str(download_dir)], check=True)
    for candidate in download_dir.glob("*.zip"):
        return candidate
    raise FileNotFoundError("No images.zip found under /kaggle/input and Kaggle dataset download failed.")


def safe_extract(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".extracted"
    archive_signature = {
        "archive": str(source.resolve()),
        "size": source.stat().st_size,
        "sha256": sha256_file(source),
    }
    if marker.exists():
        try:
            cached_signature = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Invalid extraction marker at {marker}; use an empty DATA_ROOT") from exc
        if cached_signature == archive_signature:
            return destination
        raise RuntimeError(f"DATA_ROOT was extracted from a different archive: {destination}")
    if any(destination.iterdir()):
        raise RuntimeError(f"DATA_ROOT is non-empty but has no valid extraction marker: {destination}")
    with zipfile.ZipFile(source) as archive:
        for item in archive.infolist():
            name = item.filename.replace("\\", "/")
            if name.startswith("__MACOSX/") or name.endswith("/"):
                continue
            target = (destination / name).resolve()
            if destination.resolve() not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    marker.write_text(json.dumps(archive_signature, indent=2), encoding="utf-8")
    return destination


def build_manifest(data_dir: Path) -> pd.DataFrame:
    rows = []
    paths = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for index, path in enumerate(paths):
        relative = path.relative_to(data_dir).as_posix()
        parts = relative.split("/")
        if len(parts) < 3 or parts[1] not in LABEL_MAP:
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            continue
        label = LABEL_MAP[parts[1]]
        rows.append(
            {
                "image_id": f"img_{index:05d}",
                "relative_path": relative,
                "absolute_path": str(path),
                "original_split": parts[0],
                "class_name": parts[1],
                "label": label,
                "label_name": "malignant" if label else "benign",
                "filename": path.name,
                "stem": path.stem,
                "width": width,
                "height": height,
                "file_size": path.stat().st_size,
                "md5": md5(path),
                "dhash": dhash(path),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("[WARN] image manifest is empty")
    return frame


def load_precomputed_manifest(data_dir: Path) -> pd.DataFrame | None:
    explicit = os.environ.get("MANIFEST_PATH")
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"MANIFEST_PATH is set but the active manifest is missing: {path}"
            )
    else:
        candidates = [
            ROOT / "manifest_raw.csv",
            Path("/kaggle/working/manifest_raw.csv"),
            Path("/tmp/repro_locked_assets/manifest_raw.csv"),
        ]
        for search_root in [Path("/kaggle/input"), Path("/kaggle/working")]:
            if search_root.exists():
                candidates.extend(search_root.rglob("manifest_raw.csv"))
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return None
    frame = pd.read_csv(path)
    required = {
        "image_id", "relative_path", "canonical_relative_path", "original_split",
        "label", "label_name", "filename", "md5", "dhash", "file_size", "patient_group",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Precomputed manifest is missing required columns: {sorted(missing)}")
    canonical_root = os.environ.get("CANONICAL_IMAGE_ROOT")
    if canonical_root:
        if "canonical_relative_path" not in frame.columns:
            raise ValueError("CANONICAL_IMAGE_ROOT requires canonical_relative_path in MANIFEST_PATH")
        frame["absolute_path"] = frame["canonical_relative_path"].map(
            lambda value: str(Path(canonical_root) / Path(str(value)))
        )
    else:
        frame["absolute_path"] = frame["relative_path"].map(lambda value: str(data_dir / str(value).replace("/", os.sep)))
    if not frame["absolute_path"].map(lambda value: Path(value).is_file()).all():
        raise FileNotFoundError("The precomputed manifest points to images that are absent after extraction")
    if frame["image_id"].duplicated().any() or frame["relative_path"].duplicated().any():
        raise ValueError("The precomputed manifest does not provide one-to-one image IDs and relative paths")
    if os.environ.get("VERIFY_MANIFEST_HASHES", "1").strip().lower() in {"1", "true", "yes"}:
        mismatches = []
        for row in frame.itertuples(index=False):
            observed = md5(Path(row.absolute_path))
            if observed != str(row.md5):
                mismatches.append(str(row.relative_path))
                if len(mismatches) >= 10:
                    break
        if mismatches:
            raise RuntimeError(f"Extracted archive content does not match manifest MD5 for: {mismatches}")
    return frame


def ensure_locked_assets() -> None:
    split_candidates = []
    manifest_candidates = []
    if os.environ.get("LOCKED_SPLIT"):
        split_candidates.append(Path(os.environ["LOCKED_SPLIT"]))
    if os.environ.get("MANIFEST_PATH"):
        manifest_candidates.append(Path(os.environ["MANIFEST_PATH"]))
    split_candidates.extend([ROOT / "locked_split.csv", Path("/kaggle/working/locked_split.csv"), Path("/tmp/repro_locked_assets/locked_split.csv")])
    manifest_candidates.extend([ROOT / "manifest_raw.csv", Path("/kaggle/working/manifest_raw.csv"), Path("/tmp/repro_locked_assets/manifest_raw.csv")])
    if any(path.is_file() for path in split_candidates) and any(path.is_file() for path in manifest_candidates):
        return
    target = Path("/tmp/repro_locked_assets")
    target.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for slug in ["vsdavsdvmkk/sycnn-locked-patient-split-public", "vsdavsdvmkk/sycnn-locked-patient-split", "jiaronghhh/sycnn-locked-patient-split"]:
        try:
            result = subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(target)], check=True, capture_output=True, text=True)
            diagnostics.append(f"{slug}: {result.stdout[-2000:]}")
        except Exception as exc:
            diagnostics.append(f"{slug}: {exc}")
            continue
        archives = sorted(target.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not archives:
            continue
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(target)
        if (target / "locked_split.csv").exists() and (target / "manifest_raw.csv").exists():
            return
    (target / "download_diagnostics.txt").write_text("\n\n".join(diagnostics), encoding="utf-8")
    raise FileNotFoundError("Could not download the locked patient split dataset from either account.")


_PATIENT_SUFFIX = [
    re.compile(r"^(?P<base>.+?)[ _-]\((?P<idx>\d{1,2})\)$", re.I),
    re.compile(r"^(?P<base>.+?)\((?P<idx>\d{1,2})\)$", re.I),
    re.compile(r"^(?P<base>.+?)[ _-](?P<idx>\d{1,2})$", re.I),
]


def infer_patient_group(filename: str) -> str:
    stem = re.sub(r"\s+", " ", Path(str(filename)).stem.strip().lower())
    changed = True
    while changed:
        changed = False
        for pattern in _PATIENT_SUFFIX:
            match = pattern.match(stem)
            if match and match.group("base").strip(" _-"):
                stem = match.group("base").strip(" _-")
                changed = True
                break
    return stem


def add_group_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_normalized_patient_key"] = out["filename"].map(infer_patient_group)
    parent = {value: value for value in out["image_id"]}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for column in ["md5", "filename", "dhash", "_normalized_patient_key"]:
        for _, group in out.groupby(column, dropna=False):
            ids = group["image_id"].tolist()
            for image_id in ids[1:]:
                union(ids[0], image_id)
    roots = {image_id: find(image_id) for image_id in out["image_id"]}
    anchors = {}
    grouped = out.assign(_root=out["image_id"].map(roots)).groupby("_root")
    for root, group in grouped:
        anchors[root] = sorted(group["_normalized_patient_key"].astype(str))[0]
    out["patient_group"] = out["image_id"].map(lambda value: anchors[roots[value]])
    return out.drop(columns=["_normalized_patient_key"])


def assign_groups(groups: pd.DataFrame, fractions: dict[str, float], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    work = groups.copy()
    work["tie"] = rng.random(len(work))
    work = work.sort_values(["n_images", "mixed_labels", "tie"], ascending=[False, False, True])
    totals = {key: float(work[key].sum()) for key in ["n_images", "benign", "malignant"]}
    targets = {split: {key: totals[key] * fraction for key in totals} for split, fraction in fractions.items()}
    counts = {split: {key: 0.0 for key in totals} for split in fractions}
    assignments = {}
    for _, row in work.iterrows():
        def score(split):
            score_value = 0.0
            for candidate in fractions:
                for key, weight in [("n_images", 1.0), ("benign", 2.0), ("malignant", 2.0)]:
                    value = counts[candidate][key] + (float(row[key]) if candidate == split else 0.0)
                    target = max(targets[candidate][key], 1.0)
                    score_value += weight * ((value - target) / target) ** 2
            return score_value

        chosen = min(fractions, key=score)
        assignments[str(row["patient_group"])] = chosen
        for key in totals:
            counts[chosen][key] += float(row[key])
    return assignments


def make_splits(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    work = add_group_columns(frame)
    group_counts = (
        work.groupby("patient_group")
        .agg(n_images=("image_id", "size"), benign=("label", lambda x: int((x == 0).sum())), malignant=("label", lambda x: int((x == 1).sum())), mixed_labels=("label", lambda x: x.nunique() > 1))
        .reset_index()
    )
    test2_groups = set(work.loc[work["original_split"].str.lower() == "test", "patient_group"])
    pool_groups = group_counts[~group_counts["patient_group"].isin(test2_groups)].copy()
    assign = assign_groups(pool_groups, {"train": 0.8, "val": 0.1, "test1": 0.1}, seed)
    work["split"] = work["patient_group"].map(lambda group: "test2" if group in test2_groups else assign.get(group, "unassigned"))
    if (work["split"] == "unassigned").any():
        raise RuntimeError("Some patient groups were not assigned")
    overlaps = work.groupby("patient_group")["split"].nunique()
    exact_link_crossings = count_cross_split_exact_links(work)
    audit = {
        "n_images": int(len(work)),
        "n_patient_groups": int(work["patient_group"].nunique()),
        "patient_groups_crossing_splits": int((overlaps > 1).sum()),
        "exact_duplicate_links_crossing_splits": exact_link_crossings,
        "patient_id_status": "inferred_from_filename_and_duplicate_links",
        "counts": work.groupby(["split", "label_name"]).size().unstack(fill_value=0).to_dict(),
        "legacy_target_counts": None,
    }
    if audit["patient_groups_crossing_splits"] or audit["exact_duplicate_links_crossing_splits"]:
        raise RuntimeError(f"patient-group split integrity failure: {audit}")
    return work, audit


def count_cross_split_near_duplicates(frame: pd.DataFrame, threshold: int = 2) -> int:
    """Count dHash near-duplicate pairs that cross the locked split."""
    work = frame[["dhash", "split"]].copy()
    work["dhash"] = work["dhash"].fillna("").astype(str)
    work["split"] = work["split"].astype(str)
    values = work["dhash"].tolist()
    splits = work["split"].tolist()
    cross_split = 0
    for left_index, left_value in enumerate(values):
        if not left_value:
            continue
        try:
            left_hash = int(left_value, 16)
        except ValueError:
            continue
        for right_index in range(left_index + 1, len(values)):
            right_value = values[right_index]
            if not right_value or splits[left_index] == splits[right_index]:
                continue
            try:
                distance = (left_hash ^ int(right_value, 16)).bit_count()
            except ValueError:
                continue
            if distance <= threshold:
                cross_split += 1
    return cross_split


def count_cross_split_exact_links(frame: pd.DataFrame) -> int:
    """Count byte-identical image content that crosses analysis splits.

    ``filename`` and ``dhash`` collisions are retained as audit evidence, but
    they are not duplicate identities in the authoritative raw case-directory
    cohort.  Treating camera-generated filenames as patient links would merge
    unrelated cases and would incorrectly fail the frozen split preflight.
    """
    crossings = 0
    if "md5" not in frame.columns:
        return crossings
    for _, block in frame.groupby("md5", dropna=False):
        if len(block) > 1 and block["split"].nunique() > 1:
            crossings += 1
    return crossings


def load_locked_split(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reuse the already audited patient-level split; never create a new grouping here."""
    explicit = os.environ.get("LOCKED_SPLIT")
    if explicit:
        split_path = Path(explicit).resolve()
        if not split_path.is_file():
            raise FileNotFoundError(
                f"LOCKED_SPLIT is set but the active split is missing: {split_path}"
            )
    else:
        candidates = [
            ROOT / "locked_split.csv",
            Path("/kaggle/working/locked_split.csv"),
            Path("/tmp/repro_locked_assets/locked_split.csv"),
            Path("/kaggle/input/locked-split/split_70_15_15.csv"),
        ]
        for search_root in [Path("/kaggle/input"), Path("/kaggle/working"), Path("/kaggle")]:
            if search_root.exists():
                candidates.extend(search_root.rglob("locked_split.csv"))
                candidates.extend(search_root.rglob("split_70_15_15.csv"))
        split_path = next((path for path in candidates if path.is_file()), None)
    if split_path is None:
        raise FileNotFoundError("The audited locked patient split is required; no split_70_15_15.csv was found.")
    locked = pd.read_csv(split_path)
    required = {"image_id", "relative_path", "label", "label_name", "patient_group", "split"}
    missing = required.difference(locked.columns)
    if missing:
        raise ValueError(f"Locked split is missing required columns: {sorted(missing)}")
    if len(locked) != len(manifest):
        raise ValueError(f"Locked split row count {len(locked)} does not match manifest row count {len(manifest)}")
    merged = locked.merge(
        manifest[[
            "image_id", "absolute_path", "original_split", "filename", "md5", "dhash",
            "patient_group", "label", "label_name",
        ]],
        on="image_id",
        how="left",
        suffixes=("_split", "_manifest"),
        validate="one_to_one",
    )
    if merged["absolute_path"].isna().any():
        raise ValueError("Some locked split image IDs are absent from the extracted archive")
    if merged["patient_group_split"].astype(str).ne(merged["patient_group_manifest"].astype(str)).any():
        raise ValueError("locked_split patient_group values disagree with the active manifest")
    if merged["label_split"].astype(int).ne(merged["label_manifest"].astype(int)).any():
        raise ValueError("locked_split labels disagree with the active manifest")
    merged["patient_group"] = merged["patient_group_split"].astype(str)
    merged["label"] = merged["label_split"].astype(int)
    merged["label_name"] = merged["label_name_split"].astype(str)
    merged["split"] = merged["split"].astype(str).str.lower().replace({"test": "test1"})
    patient_cross = int(merged.groupby("patient_group")["split"].nunique().gt(1).sum())
    exact_link_cross = count_cross_split_exact_links(merged)
    near_duplicate_cross = count_cross_split_near_duplicates(merged)
    source_flags = merged.assign(_is_original_test=merged["original_split"].astype(str).str.lower().eq("test"))
    source_groups = set(source_flags.loc[source_flags["_is_original_test"], "patient_group"])
    audit = {
        "split_source": str(split_path),
        "split_built_from_complete_raw_manifest": True,
        "split_reused_without_regrouping": False,
        "n_images": int(len(merged)),
        "n_patient_groups": int(merged["patient_group"].nunique()),
        "patient_groups_crossing_splits": patient_cross,
        "exact_duplicate_links_embedded_in_patient_group": True,
        "exact_duplicate_links_crossing_splits": exact_link_cross,
        "patient_group_definition": (
            "authoritative manifest patient_group"
            if "patient_group" in manifest.columns
            else "raw filename convention + exact duplicate links + dHash Hamming distance <= 2 connected components"
        ),
        "near_duplicate_links_crossing_splits": near_duplicate_cross,
        "patient_id_status": "clinically_confirmed_filename_derived_deidentified_patient_key",
        "test2_available": False,
        "source_holdout_patient_groups": int(len(source_groups)),
        "source_holdout_images": int(merged["patient_group"].isin(source_groups).sum()),
        "counts": merged.groupby(["split", "label_name"]).size().unstack(fill_value=0).to_dict(),
    }
    if patient_cross or exact_link_cross or near_duplicate_cross:
        raise RuntimeError(f"The existing locked split is not group independent: {audit}")
    return merged, audit


def image_quality_from_pil(image: Image.Image) -> float:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    contrast = float(np.clip(gray.std() / 0.25, 0.0, 1.0))
    sharpness = float(np.clip(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean(), 0.0, 0.25) / 0.25)
    exposure = float(np.clip(1.0 - abs(float(gray.mean()) - 0.5) / 0.5, 0.0, 1.0))
    return float(np.clip(0.4 * contrast + 0.4 * sharpness + 0.2 * exposure, 0.0, 1.0))


def image_quality(path: str | Path) -> float:
    with Image.open(path).convert("RGB") as image:
        return image_quality_from_pil(image)


def letterbox_square(image: Image.Image, size: int, pad_mode: str = "constant") -> Image.Image:
    """Preserve anatomy by resizing proportionally and padding to a square."""
    image = image.convert("RGB")
    width, height = image.size
    if pad_mode == "center_crop_extreme":
        aspect_ratio = max(width, 1) / max(height, 1)
        if aspect_ratio > GENERATOR_EXTREME_ASPECT_RATIO or aspect_ratio < 1.0 / GENERATOR_EXTREME_ASPECT_RATIO:
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            cropped = image.crop((left, top, left + side, top + side))
            return cropped.resize((size, size), Image.Resampling.LANCZOS)
        pad_mode = "median"
    scale = min(size / max(width, 1), size / max(height, 1))
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(target, Image.Resampling.LANCZOS)
    offset = ((size - target[0]) // 2, (size - target[1]) // 2)
    if pad_mode == "reflect":
        array = np.asarray(resized)
        left, top = offset
        right, bottom = size - target[0] - left, size - target[1] - top
        padded = np.pad(array, ((top, bottom), (left, right), (0, 0)), mode="reflect")
        return Image.fromarray(padded.astype(np.uint8, copy=False), mode="RGB")
    if pad_mode == "median":
        # Reflect padding can duplicate an eye or eyelid when the source crop
        # is very wide or very tall. A per-image border median keeps the
        # original content centered without inventing a second structure.
        array = np.asarray(resized, dtype=np.uint8)
        border = np.concatenate(
            [array[0, :, :], array[-1, :, :], array[:, 0, :], array[:, -1, :]],
            axis=0,
        )
        fill = np.median(border, axis=0).round().astype(np.uint8)
        padded = np.empty((size, size, 3), dtype=np.uint8)
        padded[:, :] = fill
        left, top = offset
        padded[top:top + target[1], left:left + target[0]] = array
        return Image.fromarray(padded, mode="RGB")
    if pad_mode != "constant":
        raise ValueError(f"Unsupported letterbox padding mode: {pad_mode}")
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, offset)
    return canvas


def degrade(image: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    mode = rng.choice(["blur", "brightness", "contrast", "noise", "jpeg"])
    if mode == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.0)))
    if mode == "brightness":
        return ImageEnhance.Brightness(image).enhance(rng.uniform(0.65, 1.35))
    if mode == "contrast":
        return ImageEnhance.Contrast(image).enhance(rng.uniform(0.65, 1.35))
    if mode == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=rng.choice([40, 60, 80]))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    arr = np.asarray(image).astype(np.float32)
    arr = np.clip(arr + np.random.default_rng(seed).normal(0, 12, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def gan_classes(image_size: int):
    import torch
    import torch.nn as nn

    # The DCGAN is parameterised by power-of-two native resolutions. The final
    # classifier always receives a single downsampled 224 px image.
    n_generator_upsamples = int(math.log2(image_size)) - 2
    generator_channels = GAN_GENERATOR_CHANNEL_PROFILES[
        GAN_GENERATOR_CHANNEL_PROFILE
    ][:n_generator_upsamples]
    maximum_discriminator_blocks = int(math.log2(image_size)) - 3
    n_discriminator_blocks = maximum_discriminator_blocks
    if GAN_DISCRIMINATOR_PATCH and GAN_DISCRIMINATOR_PATCH_BLOCKS:
        if GAN_DISCRIMINATOR_PATCH_BLOCKS > maximum_discriminator_blocks:
            raise ValueError(
                "GAN_DISCRIMINATOR_PATCH_BLOCKS cannot exceed the supported discriminator depth"
            )
        n_discriminator_blocks = GAN_DISCRIMINATOR_PATCH_BLOCKS
    elif GAN_DISCRIMINATOR_PATCH_BLOCKS and not GAN_DISCRIMINATOR_MULTISCALE:
        raise ValueError("GAN_DISCRIMINATOR_PATCH_BLOCKS requires patch or multiscale mode")
    if GAN_DISCRIMINATOR_MULTISCALE and not 1 <= GAN_DISCRIMINATOR_PATCH_BLOCKS <= maximum_discriminator_blocks:
        raise ValueError("Multiscale GAN requires a valid GAN_DISCRIMINATOR_PATCH_BLOCKS value")
    discriminator_channels = [64, 128, 256, 512, 512, 512][:n_discriminator_blocks]
    if len(generator_channels) != n_generator_upsamples or len(discriminator_channels) != n_discriminator_blocks:
        raise ValueError(f"Unsupported GAN resolution: {image_size}")

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.label = nn.Embedding(2, 32)
            self.structure_condition = bool(GAN_STRUCTURE_CONDITION)
            layers = [
                nn.ConvTranspose2d(132, generator_channels[0], 4, 1, 0, bias=False),
                nn.BatchNorm2d(generator_channels[0]),
                nn.ReLU(True),
            ]
            if self.structure_condition:
                # Teacher maps are 64 x 64.  Downsample them to the 4 x 4
                # latent grid and add them after the learned noise/label
                # projection, so the map controls eye layout while the noise
                # code still controls appearance and texture.
                structure_channels = [32, 64, 128, generator_channels[0]]
                structure_layers = []
                input_channels = 3
                for output_channels in structure_channels:
                    structure_layers.extend([
                        nn.Conv2d(input_channels, output_channels, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(output_channels),
                        nn.ReLU(True),
                    ])
                    input_channels = output_channels
                self.structure_encoder = nn.Sequential(*structure_layers)
            def upsample(cin, cout):
                if os.environ.get("GAN_UPSAMPLING", "transpose") == "resize_conv":
                    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                         nn.Conv2d(cin, cout, 3, 1, 1, bias=False))
                return nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False)
            for input_channels, output_channels in zip(generator_channels, generator_channels[1:]):
                layers.extend([
                    upsample(input_channels, output_channels),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(True),
                ])
            layers.extend([upsample(generator_channels[-1], 3), nn.Tanh()])
            self.net = nn.Sequential(*layers)

        def forward(self, noise, labels, structure_maps=None):
            code = torch.cat([noise, self.label(labels).unsqueeze(-1).unsqueeze(-1)], dim=1)
            if not self.structure_condition:
                return self.net(code)
            if structure_maps is None:
                raise ValueError("Structure-conditioned GAN requires target maps")
            if structure_maps.ndim != 4 or structure_maps.shape[1] != 3:
                raise ValueError(
                    f"Expected structure maps [N,3,H,W], got {tuple(structure_maps.shape)}"
                )
            maps = structure_maps.to(dtype=code.dtype).clamp(0.0, 1.0)
            hidden = self.net[0:3](code)
            encoded = self.structure_encoder(maps)
            if encoded.shape[-2:] != hidden.shape[-2:]:
                encoded = torch.nn.functional.interpolate(
                    encoded, size=hidden.shape[-2:], mode="nearest"
                )
            hidden = hidden + encoded
            return self.net[3:](hidden)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.label = nn.Embedding(2, 32)
            layers = []
            blocks = []
            input_channels = 35 + int(GAN_HIGH_PASS_DISCRIMINATOR)

            def discriminator_conv(input_channels, output_channels, kernel_size, stride, padding):
                layer = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=False)
                return nn.utils.spectral_norm(layer) if GAN_SPECTRAL_NORM else layer

            for block_index, output_channels in enumerate(discriminator_channels):
                block_layers = [discriminator_conv(input_channels, output_channels, 4, 2, 1)]
                if block_index and not GAN_SPECTRAL_NORM:
                    block_layers.append(nn.BatchNorm2d(output_channels))
                block_layers.append(nn.LeakyReLU(0.2, inplace=True))
                blocks.append(nn.Sequential(*block_layers))
                layers.extend(block_layers)
                input_channels = output_channels
            if GAN_DISCRIMINATOR_MULTISCALE:
                # Preserve the full-depth global decision and add a local head
                # after the requested downsampling blocks (32x32 at 256px/3).
                self.global_head = discriminator_conv(input_channels, 1, 8, 1, 0)
                local_channels = discriminator_channels[GAN_DISCRIMINATOR_PATCH_BLOCKS - 1]
                self.local_head = discriminator_conv(local_channels, 1, 1, 1, 0)
                self.blocks = nn.ModuleList(blocks)
            else:
                head_kernel = 1 if GAN_DISCRIMINATOR_PATCH else 8
                layers.append(discriminator_conv(input_channels, 1, head_kernel, 1, 0))
                self.net = nn.Sequential(*layers)

        def forward(self, image, labels, return_features=False):
            label_map = self.label(labels).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, image.shape[-2], image.shape[-1])
            values = torch.cat([image, label_map], dim=1)
            if GAN_HIGH_PASS_DISCRIMINATOR:
                values = torch.cat([values, gan_high_pass(image)], dim=1)
            if GAN_DISCRIMINATOR_MULTISCALE:
                local_logits, local_features = None, None
                for block_index, block in enumerate(self.blocks, start=1):
                    values = block(values)
                    if block_index == GAN_DISCRIMINATOR_PATCH_BLOCKS:
                        local_features = values
                        local_logits = self.local_head(values).flatten(start_dim=1)
                global_logits = self.global_head(values).view(-1)
                logits = (global_logits, local_logits)
                return (logits, [local_features, values]) if return_features else logits
            logits = self.net(values)
            logits = logits.flatten(start_dim=1) if GAN_DISCRIMINATOR_PATCH else logits.view(-1)
            return (logits, [values]) if return_features else logits

    return Generator, Discriminator


def gan_discriminator_augment(values):
    """Apply small structure-preserving differentiable augmentations to D inputs."""
    import torch
    import torch.nn.functional as F

    if not GAN_DISCRIMINATOR_AUGMENT or GAN_DISCRIMINATOR_AUGMENT_PROB <= 0:
        return values
    if float(torch.rand((), device=values.device)) >= GAN_DISCRIMINATOR_AUGMENT_PROB:
        return values
    batch, _, height, width = values.shape
    max_shift = min(GAN_DISCRIMINATOR_AUGMENT_TRANSLATION, height // 8, width // 8)
    if max_shift:
        shift_y = int(torch.randint(-max_shift, max_shift + 1, (), device=values.device).item())
        shift_x = int(torch.randint(-max_shift, max_shift + 1, (), device=values.device).item())
        padded = F.pad(values, (max_shift, max_shift, max_shift, max_shift), mode="reflect")
        values = padded[
            :, :, max_shift + shift_y:max_shift + shift_y + height,
            max_shift + shift_x:max_shift + shift_x + width,
        ]
    contrast = torch.empty((batch, 1, 1, 1), device=values.device).uniform_(0.92, 1.08)
    brightness = torch.empty((batch, 1, 1, 1), device=values.device).uniform_(-0.04, 0.04)
    values = (values * contrast + brightness).clamp(-1, 1)
    flip = torch.rand((batch, 1, 1, 1), device=values.device) > 0.5
    return torch.where(flip, values.flip(-1), values)


def gan_high_pass(values):
    """Return a fixed Sobel magnitude channel for local edge discrimination."""
    import torch
    import torch.nn.functional as F

    gray = values[:, 0:1] * 0.299 + values[:, 1:2] * 0.587 + values[:, 2:3] * 0.114
    kernel_x = values.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None] / 8.0
    kernel_y = values.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])[None, None] / 8.0
    padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    edges_x = F.conv2d(padded, kernel_x)
    edges_y = F.conv2d(padded, kernel_y)
    return torch.sqrt(edges_x.square() + edges_y.square() + 1e-6)


def discriminator_feature_matching_loss(real_features, fake_features, labels):
    """Match class-conditional discriminator feature means at two spatial scales."""
    import torch
    import torch.nn.functional as F

    terms = []
    for real_map, fake_map in zip(real_features, fake_features):
        if real_map is None or fake_map is None:
            continue
        reduce_dims = tuple(range(2, real_map.ndim))
        for label in (0, 1):
            mask = labels.eq(label)
            if not mask.any():
                continue
            real_mean = real_map[mask].mean(dim=reduce_dims)
            fake_mean = fake_map[mask].mean(dim=reduce_dims)
            scale = real_mean.detach().abs().mean().clamp_min(1e-3)
            terms.append(F.l1_loss(fake_mean, real_mean.detach()) / scale)
    return torch.stack(terms).mean() if terms else fake_features[0].new_zeros(())


def discriminator_feature_variance_loss(real_features, fake_features, labels):
    """Match class-conditional feature variance so mean matching cannot collapse modes."""
    import torch
    import torch.nn.functional as F

    terms = []
    for real_map, fake_map in zip(real_features, fake_features):
        if real_map is None or fake_map is None:
            continue
        real_vectors = real_map.flatten(2).mean(dim=2)
        fake_vectors = fake_map.flatten(2).mean(dim=2)
        for label in (0, 1):
            mask = labels.eq(label)
            if int(mask.sum()) < 2:
                continue
            real_variance = real_vectors[mask].var(dim=0, unbiased=False)
            fake_variance = fake_vectors[mask].var(dim=0, unbiased=False)
            scale = real_variance.detach().abs().mean().clamp_min(1e-3)
            terms.append(F.l1_loss(fake_variance, real_variance.detach()) / scale)
    return torch.stack(terms).mean() if terms else fake_features[0].new_zeros(())


def gan_laplacian_sharpness(values):
    """Return per-image Laplacian energy at the generator audit resolution."""
    import torch
    values = ((values + 1) / 2).clamp(0, 1)
    values = torch.nn.functional.interpolate(
        values, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
    )
    gray = values[:, 0:1] * 0.299 + values[:, 1:2] * 0.587 + values[:, 2:3] * 0.114
    kernel = gray.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])[None, None]
    return torch.nn.functional.conv2d(gray, kernel, padding=1).abs().flatten(1).mean(1)


def gan_laplacian_matching_loss(real, fake, labels):
    """Match class-conditional mean Laplacian energy without paired targets."""
    import torch
    with torch.no_grad():
        real_energy = gan_laplacian_sharpness(real)
    fake_energy = gan_laplacian_sharpness(fake)
    class_losses = []
    for label in (0, 1):
        class_mask = labels.eq(label)
        if class_mask.any():
            real_mean = real_energy[class_mask].mean().clamp_min(1e-12)
            fake_mean = fake_energy[class_mask].mean().clamp_min(1e-12)
            class_losses.append((fake_mean.log() - real_mean.log()).abs())
    return torch.stack(class_losses).mean()


def gan_edge_structure_matching_loss(real, fake, labels):
    """Match class-conditional spatial Sobel statistics, not one sharpness scalar."""
    import torch
    import torch.nn.functional as F

    def edge_grid(values):
        values = ((values + 1.0) / 2.0).clamp(0, 1)
        gray = values[:, 0:1] * 0.299 + values[:, 1:2] * 0.587 + values[:, 2:3] * 0.114
        kernel_x = values.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None] / 8.0
        kernel_y = values.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])[None, None] / 8.0
        padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
        edges = torch.sqrt(F.conv2d(padded, kernel_x).square()
                           + F.conv2d(padded, kernel_y).square() + 1e-6)
        margin = max(2, values.shape[-1] // 16)
        if edges.shape[-1] > 2 * margin:
            edges = edges[:, :, margin:-margin, margin:-margin]
        return F.adaptive_avg_pool2d(edges, (16, 16))

    with torch.no_grad():
        real_grid = edge_grid(real)
    fake_grid = edge_grid(fake)
    terms = []
    for label in (0, 1):
        mask = labels.eq(label)
        if int(mask.sum()) < 2:
            continue
        real_values = real_grid[mask]
        fake_values = fake_grid[mask]
        real_mean = real_values.mean(dim=0)
        fake_mean = fake_values.mean(dim=0)
        real_variance = real_values.var(dim=0, unbiased=False)
        fake_variance = fake_values.var(dim=0, unbiased=False)
        mean_scale = real_mean.detach().abs().mean().clamp_min(1e-3)
        variance_scale = real_variance.detach().abs().mean().clamp_min(1e-4)
        terms.append(
            F.smooth_l1_loss(fake_mean, real_mean.detach()) / mean_scale
            + 0.25 * F.smooth_l1_loss(fake_variance, real_variance.detach()) / variance_scale
        )
    return torch.stack(terms).mean() if terms else fake_grid.new_zeros(())


def train_gan(train_frame: pd.DataFrame, seed: int) -> Path:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    seed_everything(seed)
    quality_enabled = os.environ.get(
        "GAN_QUALITY_MONITOR", "resnet18_mmd" if GAN_STRUCTURE_FIRST else "equilibrium"
    ) == "resnet18_mmd"
    patience_interval_epochs = GAN_PATIENCE_INTERVAL_EPOCHS
    if GAN_OBJECTIVE == "hinge" and not quality_enabled:
        raise ValueError("Hinge GAN training requires the fixed-noise training-only quality monitor for checkpoint selection")
    source_train = train_frame
    monitor_frame = None
    if quality_enabled:
        from generator_quality import split_generator_training
        train_frame, monitor_frame = split_generator_training(source_train, seed)
    if GAN_EYE_STRUCTURE_SELECTION_WEIGHT and not quality_enabled:
        raise ValueError("GAN eye-structure checkpoint selection requires the fixed quality monitor")
    device = torch.device(device_name())
    eye_teacher = None
    eye_target_cache = None
    eye_reference = {}
    eye_target_pools = {}
    eye_sampler_rng = None
    eye_target_orders = {}
    eye_target_cursors = {0: 0, 1: 0}
    teacher_sha256 = None
    target_cache_sha256 = None
    if GAN_STRUCTURE_CONDITION or GAN_EYE_STRUCTURE_LOSS_WEIGHT or GAN_EYE_STRUCTURE_SELECTION_WEIGHT:
        from eye_structure_prior import EyeTargetCache, PretrainedEyeTeacher, IRIS_CHANNEL, EYELID_CHANNEL

        teacher_path = Path(os.environ["GAN_EYE_TEACHER_WEIGHTS"]).resolve(strict=True)
        target_cache_path = Path(os.environ["GAN_EYE_TARGET_CACHE"]).resolve(strict=True)
        teacher_sha256 = sha256_file(teacher_path)
        target_cache_sha256 = sha256_file(target_cache_path)
        eye_teacher = PretrainedEyeTeacher(teacher_path, device)
        eye_target_cache = EyeTargetCache(target_cache_path, expected_weights_sha256=teacher_sha256)
        if GAN_EYE_STRUCTURE_SELECTION_WEIGHT:
            eye_reference = eye_target_cache.reference_statistics(monitor_frame)
            for label in (0, 1):
                if eye_reference[label].get("eye_presence_rate", 0.0) <= 0:
                    raise RuntimeError(f"Patient-disjoint GAN monitor has no trusted class-{label} eye targets")
        if GAN_EYE_STRUCTURE_LOSS_WEIGHT:
            fit_indices = np.asarray([
                eye_target_cache.index[str(image_id)] for image_id in train_frame["image_id"].astype(str)
            ], dtype=np.int64)
            fit_labels = train_frame["label"].astype(int).to_numpy()
            if not np.array_equal(eye_target_cache.labels[fit_indices], fit_labels):
                raise ValueError("GAN fit labels do not align with the train-only eye-target cache")
            active_targets = eye_target_cache.active[fit_indices]
            trusted_pairs = active_targets[:, IRIS_CHANNEL] & active_targets[:, EYELID_CHANNEL]
            for label in (0, 1):
                positions = np.flatnonzero((fit_labels == label) & trusted_pairs)
                if not len(positions):
                    raise RuntimeError(f"GAN fit partition has no trusted class-{label} eye masks")
                selected_indices = fit_indices[positions]
                eye_target_pools[label] = (
                    torch.from_numpy(eye_target_cache.targets[selected_indices].copy()),
                    torch.from_numpy(eye_target_cache.active[selected_indices].copy()),
                )
            eye_sampler_rng = np.random.default_rng(seed + 271829)
            eye_target_orders = {
                label: eye_sampler_rng.permutation(len(eye_target_pools[label][0]))
                for label in (0, 1)
            }
    if GAN_STRUCTURE_CONDITION and eye_sampler_rng is None:
        raise ValueError(
            "GAN_STRUCTURE_CONDITION requires a train-only eye target sampler"
        )

    def sample_eye_targets(labels):
        if eye_sampler_rng is None:
            raise RuntimeError("GAN eye targets were requested without a configured eye prior")
        chosen_targets, chosen_active = [], []
        for label in labels.detach().cpu().tolist():
            label = int(label)
            targets, active = eye_target_pools[label]
            if eye_target_cursors[label] >= len(eye_target_orders[label]):
                eye_target_orders[label] = eye_sampler_rng.permutation(len(targets))
                eye_target_cursors[label] = 0
            index = int(eye_target_orders[label][eye_target_cursors[label]])
            eye_target_cursors[label] += 1
            chosen_targets.append(targets[index])
            chosen_active.append(active[index])
        return torch.stack(chosen_targets).to(device), torch.stack(chosen_active).to(device)

    output = OUT / "synthetic" / f"seed_{seed}"
    marker = output / "gan_complete.json"
    output.mkdir(parents=True, exist_ok=True)
    gan_configuration = {
        "seed": seed,
        "gan_size": GAN_SIZE,
        "gan_epochs": GAN_EPOCHS,
        "gan_batch_size": GAN_BATCH_SIZE,
        "gan_patience": GAN_PATIENCE,
        "gan_patience_interval_epochs": patience_interval_epochs,
        "gan_early_stopping_patience_unit": (
            f"{patience_interval_epochs}-epoch_non_improvement_blocks"
            if patience_interval_epochs > 1 else "epochs"
        ),
        "gan_min_epochs": GAN_MIN_EPOCHS,
        "gan_min_delta": GAN_MIN_DELTA,
        "gan_structure_first": GAN_STRUCTURE_FIRST,
        "gan_objective": GAN_OBJECTIVE,
        "gan_spectral_norm": GAN_SPECTRAL_NORM,
        "gan_discriminator_patch": GAN_DISCRIMINATOR_PATCH,
        "gan_discriminator_multiscale": GAN_DISCRIMINATOR_MULTISCALE,
        "gan_discriminator_patch_blocks": (
            GAN_DISCRIMINATOR_PATCH_BLOCKS or int(math.log2(GAN_SIZE)) - 3
            if GAN_DISCRIMINATOR_PATCH or GAN_DISCRIMINATOR_MULTISCALE else None
        ),
        "gan_discriminator_batch_norm": not GAN_SPECTRAL_NORM,
        "gan_sharpness_selection_weight": GAN_SHARPNESS_SELECTION_WEIGHT,
        "gan_laplacian_match_weight": GAN_LAPLACIAN_MATCH_WEIGHT,
        "gan_feature_match_weight": GAN_FEATURE_MATCH_WEIGHT,
        "gan_feature_match_variance_weight": GAN_FEATURE_MATCH_VARIANCE_WEIGHT,
        "gan_edge_structure_match_weight": GAN_EDGE_STRUCTURE_MATCH_WEIGHT,
        "gan_eye_structure_loss_weight": GAN_EYE_STRUCTURE_LOSS_WEIGHT,
        "gan_eye_structure_selection_weight": GAN_EYE_STRUCTURE_SELECTION_WEIGHT,
        "gan_structure_condition": GAN_STRUCTURE_CONDITION,
        "structure_condition_channels": 3 if GAN_STRUCTURE_CONDITION else 0,
        "structure_condition_source": (
            "train_eye_targets_iris_eyelid_trusted" if GAN_STRUCTURE_CONDITION else None
        ),
        "eye_teacher_weights_sha256": teacher_sha256,
        "eye_target_cache_sha256": target_cache_sha256,
        "gan_discriminator_augment": GAN_DISCRIMINATOR_AUGMENT,
        "gan_discriminator_augment_prob": GAN_DISCRIMINATOR_AUGMENT_PROB,
        "gan_discriminator_augment_translation": GAN_DISCRIMINATOR_AUGMENT_TRANSLATION,
        "gan_high_pass_discriminator": GAN_HIGH_PASS_DISCRIMINATOR,
        "gan_early_stopping_enabled": bool(GAN_MIN_EPOCHS < GAN_EPOCHS or GAN_PATIENCE < GAN_EPOCHS),
        "batch_size": GAN_BATCH_SIZE,
        "pool_per_class": int(os.environ.get("GAN_POOL_PER_CLASS", "2048")),
        "train_image_ids_sha256": hashlib.sha256(
            "\n".join(sorted(train_frame["image_id"].astype(str))).encode("utf-8")
        ).hexdigest(),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "upsampling": os.environ.get("GAN_UPSAMPLING", "transpose"),
        "generator_channel_profile": GAN_GENERATOR_CHANNEL_PROFILE,
        "generator_channels": GAN_GENERATOR_CHANNEL_PROFILES[
            GAN_GENERATOR_CHANNEL_PROFILE
        ][: int(math.log2(GAN_SIZE)) - 2],
        "padding_mode": GENERATOR_PAD_MODE,
        "extreme_aspect_ratio": GENERATOR_EXTREME_ASPECT_RATIO,
        "quality_monitor": "resnet18_mmd" if quality_enabled else "equilibrium",
        "ema_decay": float(os.environ.get("GAN_EMA_DECAY", "0.999")),
        "monitor_samples_per_class": int(os.environ.get("GAN_MONITOR_SAMPLES", "64")),
        "quality_code_sha256": sha256_file(ROOT / "generator_quality.py") if quality_enabled else None,
        "monitor_image_ids_sha256": hashlib.sha256("\n".join(sorted(monitor_frame["image_id"].astype(str))).encode()).hexdigest() if quality_enabled else None,
    }
    gan_fingerprint = hashlib.sha256(json.dumps(gan_configuration, sort_keys=True).encode("utf-8")).hexdigest()
    if marker.exists() and not FORCE_RERUN:
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        if marker_payload.get("configuration_fingerprint") != gan_fingerprint:
            raise RuntimeError(f"Existing GAN output uses another data/code/settings fingerprint: {output}")
        pool_files = list(output.glob("pool_class*.png"))
        history_path = output / "gan_history.csv"
        if (
            marker_payload.get("configuration_fingerprint") == gan_fingerprint
            and len(pool_files) == 2 * gan_configuration["pool_per_class"]
            and history_path.is_file() and history_path.stat().st_size > 0
        ):
            return output
    transform = transforms.Compose([
        transforms.Lambda(lambda image: letterbox_square(image, GAN_SIZE, pad_mode=GENERATOR_PAD_MODE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    class RealDataset(Dataset):
        def __len__(self):
            return len(train_frame)

        def __getitem__(self, index):
            row = train_frame.iloc[index]
            with Image.open(row["absolute_path"]).convert("RGB") as image:
                tensor = transform(image)
            return tensor, int(row["label"])

    loader = DataLoader(RealDataset(), batch_size=GAN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
    Generator, Discriminator = gan_classes(GAN_SIZE)
    generator, discriminator = Generator().to(device), Discriminator().to(device)
    quality_monitor, averaged_generator = None, None
    if quality_enabled:
        from generator_quality import GanQualityMonitor, ema_model, update_ema
        monitor_structure_conditions = None
        if GAN_STRUCTURE_CONDITION:
            monitor_structure_conditions = {}
            monitor_count = gan_configuration["monitor_samples_per_class"]
            for label in (0, 1):
                rows = monitor_frame.loc[monitor_frame["label"].astype(int).eq(label)].sort_values("image_id")
                maps = []
                for _, row in rows.iterrows():
                    index = eye_target_cache.index[str(row["image_id"])]
                    active = eye_target_cache.active[index]
                    if bool(active[IRIS_CHANNEL] and active[EYELID_CHANNEL]):
                        target = eye_target_cache.targets[index].copy()
                        target *= active[:, None, None].astype(np.float32)
                        maps.append(torch.from_numpy(target))
                if not maps:
                    raise RuntimeError(
                        f"GAN structure-conditioned monitor has no trusted class-{label} maps"
                    )
                order = np.arange(len(maps), dtype=int)
                np.random.default_rng(seed + 7919 + label * 104729).shuffle(order)
                maps = [maps[int(order[index % len(order)])] for index in range(monitor_count)]
                monitor_structure_conditions[label] = torch.stack(maps)
        quality_monitor = GanQualityMonitor(monitor_frame, transform, device, seed,
                                            gan_configuration["monitor_samples_per_class"],
                                            structure_conditions=monitor_structure_conditions)
        averaged_generator = ema_model(generator)
        pd.concat([train_frame.assign(generator_partition="fit"),
                   monitor_frame.assign(generator_partition="monitor")]).to_csv(output / "gan_training_partition.csv", index=False)
    criterion = torch.nn.BCEWithLogitsLoss() if GAN_OBJECTIVE == "bce" else None
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    checkpoint_path = output / "gan_checkpoint.pt"
    start_epoch = 0
    best_monitor = np.inf
    best_epoch = 0
    best_generator_state = None
    best_discriminator_state = None
    stale = 0
    history = []
    optimizer_updates = 0
    if checkpoint_path.exists() and not FORCE_RERUN:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("configuration_fingerprint") != gan_fingerprint:
            raise RuntimeError(f"GAN checkpoint does not match this run; use a new output root: {checkpoint_path}")
        if checkpoint.get("configuration_fingerprint") == gan_fingerprint:
            generator.load_state_dict(checkpoint["generator"])
            discriminator.load_state_dict(checkpoint["discriminator"])
            g_optimizer.load_state_dict(checkpoint["g_optimizer"])
            d_optimizer.load_state_dict(checkpoint["d_optimizer"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_monitor = float(checkpoint.get("best_monitor", np.inf))
            best_epoch = int(checkpoint.get("best_epoch", 0))
            best_generator_state = checkpoint.get("best_generator_state")
            best_discriminator_state = checkpoint.get("best_discriminator_state")
            stale = int(checkpoint.get("stale", 0))
            history = list(checkpoint.get("history", []))
            optimizer_updates = int(checkpoint.get("optimizer_updates", 0))
            if averaged_generator is not None:
                averaged_generator.load_state_dict(checkpoint["ema_generator"])
            if eye_sampler_rng is not None:
                sampler_state = checkpoint.get("eye_target_sampler")
                if sampler_state is None:
                    raise RuntimeError("GAN checkpoint is missing its train-only eye-target sampler state")
                eye_sampler_rng.bit_generator.state = sampler_state["rng_state"]
                eye_target_orders = {int(label): order for label, order in sampler_state["orders"].items()}
                eye_target_cursors = {int(label): int(value) for label, value in sampler_state["cursors"].items()}
            restore_rng_state(checkpoint["rng_state"])
            print(f"[RESUME] GAN seed={seed} from epoch {start_epoch}")
    if start_epoch >= GAN_MIN_EPOCHS and stale >= GAN_PATIENCE:
        start_epoch = GAN_EPOCHS  # Resume generation, not another stopped training epoch.
    for epoch in range(start_epoch, GAN_EPOCHS):
        generator.train()
        discriminator.train()
        discriminator_loss_sum = 0.0
        generator_loss_sum = 0.0
        laplacian_loss_sum = 0.0
        feature_matching_loss_sum = 0.0
        feature_variance_loss_sum = 0.0
        edge_structure_loss_sum = 0.0
        eye_structure_loss_sum = 0.0
        batches = 0
        for real, labels in loader:
            real, labels = real.to(device), labels.to(device)
            noise = torch.randn(len(real), 100, 1, 1, device=device)
            batch_eye_targets, batch_eye_active = None, None
            if GAN_STRUCTURE_CONDITION or GAN_EYE_STRUCTURE_LOSS_WEIGHT:
                batch_eye_targets, batch_eye_active = sample_eye_targets(labels)
            fake = generator(
                noise,
                labels,
                batch_eye_targets * batch_eye_active[:, :, None, None].to(dtype=noise.dtype)
                if GAN_STRUCTURE_CONDITION else None,
            )
            d_optimizer.zero_grad(set_to_none=True)
            real_logits = discriminator(gan_discriminator_augment(real), labels)
            fake_logits = discriminator(gan_discriminator_augment(fake.detach()), labels)
            if GAN_DISCRIMINATOR_MULTISCALE:
                real_global, real_local = real_logits
                fake_global, fake_local = fake_logits
                if GAN_OBJECTIVE == "hinge":
                    discriminator_loss = 0.5 * (
                        torch.nn.functional.relu(1.0 - real_global).mean()
                        + torch.nn.functional.relu(1.0 + fake_global).mean()
                        + torch.nn.functional.relu(1.0 - real_local).mean()
                        + torch.nn.functional.relu(1.0 + fake_local).mean()
                    )
                else:
                    discriminator_loss = 0.5 * (
                        criterion(real_global, torch.ones_like(real_global))
                        + criterion(fake_global, torch.zeros_like(fake_global))
                        + criterion(real_local, torch.ones_like(real_local))
                        + criterion(fake_local, torch.zeros_like(fake_local))
                    )
            elif GAN_OBJECTIVE == "hinge":
                discriminator_loss = (
                    torch.nn.functional.relu(1.0 - real_logits).mean()
                    + torch.nn.functional.relu(1.0 + fake_logits).mean()
                )
            else:
                real_loss = criterion(real_logits, torch.ones_like(real_logits))
                fake_loss = criterion(fake_logits, torch.zeros_like(fake_logits))
                discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            d_optimizer.step()
            g_optimizer.zero_grad(set_to_none=True)
            generator_output = discriminator(
                gan_discriminator_augment(fake), labels, return_features=True
            )
            generator_logits, fake_features = generator_output
            with torch.no_grad():
                _, real_features = discriminator(
                    gan_discriminator_augment(real), labels, return_features=True
                )
            if GAN_DISCRIMINATOR_MULTISCALE:
                generator_global, generator_local = generator_logits
                if GAN_OBJECTIVE == "hinge":
                    g_loss = -0.5 * (generator_global.mean() + generator_local.mean())
                else:
                    g_loss = 0.5 * (
                        criterion(generator_global, torch.ones_like(generator_global))
                        + criterion(generator_local, torch.ones_like(generator_local))
                    )
            elif GAN_OBJECTIVE == "hinge":
                g_loss = -generator_logits.mean()
            else:
                g_loss = criterion(generator_logits, torch.ones_like(generator_logits))
            feature_matching_loss = discriminator_feature_matching_loss(
                real_features, fake_features, labels
            )
            if GAN_FEATURE_MATCH_WEIGHT:
                g_loss = g_loss + GAN_FEATURE_MATCH_WEIGHT * feature_matching_loss
            feature_variance_loss = discriminator_feature_variance_loss(
                real_features, fake_features, labels
            )
            if GAN_FEATURE_MATCH_VARIANCE_WEIGHT:
                g_loss = g_loss + GAN_FEATURE_MATCH_VARIANCE_WEIGHT * feature_variance_loss
            edge_structure_loss = None
            if GAN_EDGE_STRUCTURE_MATCH_WEIGHT:
                edge_structure_loss = gan_edge_structure_matching_loss(real, fake, labels)
                g_loss = g_loss + GAN_EDGE_STRUCTURE_MATCH_WEIGHT * edge_structure_loss
            laplacian_loss = None
            if GAN_LAPLACIAN_MATCH_WEIGHT:
                laplacian_loss = gan_laplacian_matching_loss(real, fake, labels)
                g_loss = g_loss + GAN_LAPLACIAN_MATCH_WEIGHT * laplacian_loss
            eye_structure_loss = None
            if GAN_EYE_STRUCTURE_LOSS_WEIGHT:
                eye_structure_loss = eye_teacher.loss(
                    fake, batch_eye_targets, batch_eye_active,
                    torch.ones(len(labels), device=device, dtype=fake.dtype),
                )
                g_loss = g_loss + GAN_EYE_STRUCTURE_LOSS_WEIGHT * eye_structure_loss
            g_loss.backward()
            g_optimizer.step()
            optimizer_updates += 1
            if averaged_generator is not None:
                update_ema(averaged_generator.state_dict(), generator, gan_configuration["ema_decay"], optimizer_updates)
            discriminator_loss_sum += float(discriminator_loss.item())
            generator_loss_sum += float(g_loss.item())
            feature_matching_loss_sum += float(feature_matching_loss.detach().item())
            feature_variance_loss_sum += float(feature_variance_loss.detach().item())
            if edge_structure_loss is not None:
                edge_structure_loss_sum += float(edge_structure_loss.detach().item())
            if laplacian_loss is not None:
                laplacian_loss_sum += float(laplacian_loss.detach().item())
            if eye_structure_loss is not None:
                eye_structure_loss_sum += float(eye_structure_loss.detach().item())
            batches += 1
        mean_discriminator_loss = discriminator_loss_sum / max(batches, 1)
        mean_generator_loss = generator_loss_sum / max(batches, 1)
        mean_feature_matching_loss = feature_matching_loss_sum / max(batches, 1)
        mean_feature_variance_loss = feature_variance_loss_sum / max(batches, 1)
        mean_edge_structure_loss = (
            edge_structure_loss_sum / max(batches, 1) if GAN_EDGE_STRUCTURE_MATCH_WEIGHT else None
        )
        mean_laplacian_loss = (
            laplacian_loss_sum / max(batches, 1) if GAN_LAPLACIAN_MATCH_WEIGHT else None
        )
        mean_eye_structure_loss = (
            eye_structure_loss_sum / max(batches, 1) if GAN_EYE_STRUCTURE_LOSS_WEIGHT else 0.0
        )
        # BCE equilibrium is D ~= 2*log(2), G ~= log(2). Hinge runs use the
        # patient-disjoint training monitor for selection instead.
        equilibrium = (
            abs(mean_discriminator_loss - 2.0 * math.log(2.0))
            + abs(mean_generator_loss - math.log(2.0))
            if GAN_OBJECTIVE == "bce" else None
        )
        monitor, quality_details, preview = equilibrium, {}, None
        if quality_monitor is not None:
            monitor, quality_details, preview = quality_monitor.evaluate(averaged_generator)
            eye_structure_penalty = 0.0
            if eye_teacher is not None and GAN_EYE_STRUCTURE_SELECTION_WEIGHT:
                class_scores = []
                for label in (0, 1):
                    score, details = eye_teacher.generated_quality(
                        quality_monitor.last_generated[label].to(device), eye_reference[label]
                    )
                    class_scores.append(score)
                    quality_details.update({
                        f"quality_class_{label}_{key}": value for key, value in details.items()
                    })
                eye_structure_penalty = float(np.mean(class_scores))
            quality_details["quality_eye_structure_penalty"] = eye_structure_penalty
            quality_details["quality_eye_structure_selection_penalty"] = (
                GAN_EYE_STRUCTURE_SELECTION_WEIGHT * eye_structure_penalty
            )
            quality_details["quality_mmd2_raw"] = float(monitor)
            monitor = float(monitor + GAN_EYE_STRUCTURE_SELECTION_WEIGHT * eye_structure_penalty)
            if GAN_SHARPNESS_SELECTION_WEIGHT:
                sharpness_log_distance = float(np.mean([
                    abs(math.log(max(quality_details[f"sharpness_generated_class_{label}"], 1e-12)
                                 / max(quality_details[f"sharpness_real_class_{label}"], 1e-12)))
                    for label in (0, 1)
                ]))
                quality_details["mmd_selection_score"] = float(monitor)
                quality_details["sharpness_log_distance"] = sharpness_log_distance
                quality_details["sharpness_selection_penalty"] = (
                    GAN_SHARPNESS_SELECTION_WEIGHT * sharpness_log_distance
                )
                monitor = float(monitor + quality_details["sharpness_selection_penalty"])
        if monitor < best_monitor - GAN_MIN_DELTA:
            best_monitor = monitor
            best_epoch = epoch + 1
            selected_generator = averaged_generator if averaged_generator is not None else generator
            best_generator_state = {key: value.detach().cpu().clone() for key, value in selected_generator.state_dict().items()}
            best_discriminator_state = {key: value.detach().cpu().clone() for key, value in discriminator.state_dict().items()}
            stale = 0
            if preview is not None:
                from torchvision.utils import save_image
                save_image((preview + 1) / 2, output / "best_fixed_examples.png", nrow=4)
        else:
            epochs_since_improvement = epoch + 1 - best_epoch
            stale = epochs_since_improvement // patience_interval_epochs
        history.append({
            "epoch": epoch + 1,
            "discriminator_loss": mean_discriminator_loss,
            "generator_loss": mean_generator_loss,
            "feature_matching_loss": mean_feature_matching_loss,
            "feature_variance_loss": mean_feature_variance_loss,
            "edge_structure_match_loss": mean_edge_structure_loss,
            "laplacian_match_loss": mean_laplacian_loss,
            "eye_structure_loss": mean_eye_structure_loss,
            "equilibrium_distance": equilibrium,
            "selection_score": monitor,
            "best_selection_score": best_monitor,
            **quality_details,
            "early_stop_stale_units": stale,
        })
        laplacian_log = (
            f" laplacian_loss={mean_laplacian_loss:.4f}"
            if mean_laplacian_loss is not None else ""
        )
        print(
            f"[GAN] seed={seed} epoch={epoch + 1}/{GAN_EPOCHS} "
            f"d_loss={mean_discriminator_loss:.4f} g_loss={mean_generator_loss:.4f}"
            f" eye={mean_eye_structure_loss:.4f}"
            f"{laplacian_log} "
            f"selection_score={monitor:.6f} stale_units={stale}/{GAN_PATIENCE} "
            f"unit={gan_configuration['gan_early_stopping_patience_unit']}"
        )
        checkpoint = {
            "configuration_fingerprint": gan_fingerprint,
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "g_optimizer": g_optimizer.state_dict(),
            "d_optimizer": d_optimizer.state_dict(),
            "best_monitor": best_monitor,
            "best_epoch": best_epoch,
            "best_generator_state": best_generator_state,
            "best_discriminator_state": best_discriminator_state,
            "stale": stale,
            "history": history,
            "optimizer_updates": optimizer_updates,
            "ema_generator": averaged_generator.state_dict() if averaged_generator is not None else None,
            "eye_target_sampler": (
                {
                    "rng_state": eye_sampler_rng.bit_generator.state,
                    "orders": eye_target_orders,
                    "cursors": eye_target_cursors,
                } if eye_sampler_rng is not None else None
            ),
            "rng_state": capture_rng_state(),
        }
        temporary_checkpoint = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary_checkpoint)
        temporary_checkpoint.replace(checkpoint_path)
        if epoch + 1 >= GAN_MIN_EPOCHS and stale >= GAN_PATIENCE:
            print(
                f"[GAN EARLY STOP] seed={seed} epoch={epoch + 1} "
                f"after {stale} stale units of {gan_configuration['gan_early_stopping_patience_unit']}"
            )
            break
    if best_generator_state is None:
        best_epoch = int(history[-1]["epoch"]) if history else 0
        best_generator_state = {key: value.detach().cpu().clone() for key, value in generator.state_dict().items()}
    generator.load_state_dict(best_generator_state)
    generator.eval()
    torch.save(generator.state_dict(), output / "generator_state.pt")
    with torch.no_grad():
        pool_count = gan_configuration["pool_per_class"]
        for label in [0, 1]:
            generation_batch = int(os.environ.get("GAN_GENERATION_BATCH", "128"))
            for start in range(0, pool_count, generation_batch):
                stop = min(start + generation_batch, pool_count)
                noise = torch.randn(stop - start, 100, 1, 1, device=device)
                labels = torch.full((stop - start,), label, dtype=torch.long, device=device)
                generation_conditions = None
                if GAN_STRUCTURE_CONDITION:
                    generation_targets, generation_active = sample_eye_targets(labels)
                    generation_conditions = generation_targets * generation_active[:, :, None, None].to(dtype=noise.dtype)
                samples = generator(noise, labels, generation_conditions).cpu()
                for offset, sample in enumerate(samples):
                    index = start + offset
                    array = ((sample.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    Image.fromarray(array).save(output / f"pool_class{label}_{index:04d}.png")
    pd.DataFrame(history).to_csv(output / "gan_history.csv", index=False)
    marker.write_text(json.dumps({
        **gan_configuration,
        "configuration_fingerprint": gan_fingerprint,
        "generator_initialization": "seeded PyTorch module defaults",
        "generator_pretrained_weights_used": False,
        "frozen_pretrained_eye_teacher_used": teacher_sha256 is not None,
        "completed_epochs": int(history[-1]["epoch"]) if history else 0,
        "selected_generator_epoch": int(best_epoch),
        "selected_generator_score": float(best_monitor),
        "early_stopping_monitor": (
            f"{quality_monitor.metric_name}+weighted_pretrained_eye_structure_penalty"
            if quality_monitor is not None and GAN_EYE_STRUCTURE_SELECTION_WEIGHT
            else quality_monitor.metric_name if quality_monitor is not None
            else "training_bce_equilibrium_distance"
        ),
        "eye_structure_loss_weight": GAN_EYE_STRUCTURE_LOSS_WEIGHT,
        "eye_structure_selection_weight": GAN_EYE_STRUCTURE_SELECTION_WEIGHT,
        "eye_teacher_weights_sha256": teacher_sha256,
        "eye_target_cache_sha256": target_cache_sha256,
        "early_stopping_patience_unit": gan_configuration["gan_early_stopping_patience_unit"],
        "selected_checkpoint": "ema" if averaged_generator is not None else "raw",
        "status": "complete",
    }, indent=2), encoding="utf-8")
    # Keep the remote resumable state; local preparation never runs training.
    return output


def synthetic_frame(pool_dir: Path, real_frame: pd.DataFrame, ratio: float, strategy: str, seed: int) -> pd.DataFrame:
    if ratio <= 0:
        return pd.DataFrame(columns=["image_id", "absolute_path", "label", "label_name", "patient_group", "synthetic"])
    if pool_dir is None:
        raise RuntimeError(f"Synthetic ratio n={ratio:g} requires a completed train-only GAN pool")
    rows = []
    total = int(round(ratio * len(real_frame)))
    counts = real_frame["label"].value_counts().to_dict()
    if strategy == "balanced":
        # Balance the final real-plus-synthetic training set, rather than only
        # balancing the synthetic subset. This isolates class-balance utility.
        if (len(real_frame) + total) % 2:
            total += 1
        real_benign = int(counts.get(0, 0))
        real_malignant = int(counts.get(1, 0))
        target_counts = {
            0: (total - real_benign + real_malignant) // 2,
            1: total - (total - real_benign + real_malignant) // 2,
        }
        if min(target_counts.values()) < 0:
            raise RuntimeError("Synthetic ratio is too small to achieve final 1:1 class balance")
    else:
        target_counts = {label: int(round(total * counts.get(label, 0) / len(real_frame))) for label in [0, 1]}
        target_counts[1] += total - sum(target_counts.values())
    for label, count in target_counts.items():
        candidates = sorted(pool_dir.glob(f"pool_class{label}_*.png"))
        if not candidates:
            raise FileNotFoundError(f"No GAN pool for class {label} in {pool_dir}")
        # A ratio-independent seeded order makes the n=3, n=4, and n=5 sets
        # nested: each larger ratio adds images instead of replacing the pool.
        rng = np.random.default_rng(seed + 104729 * label)
        ordered = np.asarray(candidates, dtype=object)[rng.permutation(len(candidates))]
        if count > len(ordered):
            raise RuntimeError(
                f"GAN pool for class {label} has {len(candidates)} unique images, "
                f"but n={ratio:g} requires {count}; increase GAN_POOL_PER_CLASS"
            )
        for rank, path in enumerate(ordered[:count]):
            rows.append({"image_id": f"syn_{seed}_{strategy}_{ratio}_{label}_{rank:05d}", "absolute_path": str(path), "label": int(label), "label_name": "malignant" if label else "benign", "patient_group": f"synthetic_{seed}_{strategy}_{ratio}_{label}_{rank:05d}", "synthetic": True, "synthetic_pool_rank": int(rank)})
    return pd.DataFrame(rows)


def make_transform(size: int, mode: str, train: bool):
    from torchvision import transforms

    pad_mode = "reflect" if os.environ.get("MANUAL_EYE_DATASET_ROOT") else "constant"
    operations = [transforms.Lambda(lambda image: letterbox_square(image, size, pad_mode=pad_mode))]
    if train:
        operations.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(8)])
        if mode == "strong":
            operations.extend([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15), transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1))])
    operations.extend([transforms.ToTensor(), transforms.Normalize(RGB_MEAN, RGB_STD)])
    return transforms.Compose(operations)


def build_backbone(name: str = "resnet50", initialization: str = BACKBONE_INIT):
    import torch.nn as nn
    from torchvision import models

    normalized = name.lower()
    use_imagenet = initialization == "imagenet"
    if normalized == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if use_imagenet else None
        model = models.resnet34(weights=weights)
    elif normalized in {"efficientnet_b0", "efficientnet-b0", "efficientnetb0"}:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_imagenet else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    else:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if use_imagenet else None
        model = models.resnet50(weights=weights)
    if any(parameter.requires_grad for parameter in model.parameters()) and getattr(model, "fc", None) is not None:
        model.fc = nn.Linear(model.fc.in_features, 1)
    model._pretrained_weights_used = use_imagenet
    model._initialization = initialization
    return model


class ImageDataset:
    def __init__(self, frame: pd.DataFrame, size: int, mode: str, train: bool, quality: bool, degradation_seed: int = 0):
        from multiprocessing import Value
        import torch
        from torch.utils.data import Dataset

        # Shared state makes the stochastic quality degradation reproducible
        # across DataLoader workers while varying it across training epochs.
        self._epoch = Value("i", 0)

        class _Dataset(Dataset):
            def __len__(self_inner):
                return len(frame)

            def __getitem__(self_inner, index):
                row = frame.iloc[index]
                with Image.open(row["absolute_path"]).convert("RGB") as image:
                    if train and mode == "degradation" and not bool(row.get("synthetic", False)):
                        image = degrade(image, degradation_seed + index + self._epoch.value * 1_000_003)
                    quality_value = image_quality_from_pil(image) if quality else 1.0
                    tensor = make_transform(size, mode, train)(image)
                return tensor, torch.tensor(float(row["label"]), dtype=torch.float32), str(row["image_id"]), torch.tensor(float(quality_value), dtype=torch.float32)

        self.dataset = _Dataset()

    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = int(epoch)


def train_backbone(train_frame: pd.DataFrame, val_frame: pd.DataFrame, size: int, mode: str, quality: bool, class_weighted: bool, seed: int, backbone: str, checkpoint_dir: Path, configuration_fingerprint: str):
    import torch
    from torch.utils.data import DataLoader

    seed_everything(seed)
    device = torch.device(device_name())
    model = build_backbone(backbone, BACKBONE_INIT).to(device)
    if bool(getattr(model, "_pretrained_weights_used", False)) != (BACKBONE_INIT == "imagenet"):
        raise RuntimeError("Backbone initialization does not match BACKBONE_INIT")
    train_dataset = ImageDataset(train_frame, size, mode, True, quality, degradation_seed=seed)
    train_loader = DataLoader(train_dataset.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(ImageDataset(val_frame, size, "none", False, quality).dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    positive = float((train_frame["label"] == 1).sum())
    negative = float((train_frame["label"] == 0).sum())
    pos_weight = torch.tensor([negative / max(positive, 1.0)], device=device) if class_weighted else None
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none" if quality else "mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_state, best_score, best_epoch, stale, history = None, -np.inf, 0, 0, []
    checkpoint_path = checkpoint_dir / "backbone_checkpoint.pt"
    start_epoch = 1
    if checkpoint_path.exists() and not FORCE_RERUN:
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
            start_epoch = int(checkpoint["epoch"]) + 1
            restore_rng_state(checkpoint["rng_state"])
            print(f"[RESUME] CNN seed={seed} from epoch {start_epoch}")
            if int(checkpoint.get("epoch", 0)) >= CNN_MIN_EPOCHS and stale >= PATIENCE:
                start_epoch = EPOCHS + 1
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_dataset.set_epoch(epoch)
        total = 0.0
        for x, y, _, q in train_loader:
            x, y, q = x.to(device), y.to(device), q.to(device)
            optimizer.zero_grad(set_to_none=True)
            if mode == "strong":
                mix_lambda = float(np.random.beta(0.4, 0.4))
                permutation = torch.randperm(x.size(0), device=device)
                mixed_x = mix_lambda * x + (1.0 - mix_lambda) * x[permutation]
                logits = model(mixed_x).view(-1)
                loss = mix_lambda * loss_fn(logits, y) + (1.0 - mix_lambda) * loss_fn(logits, y[permutation])
                q = mix_lambda * q + (1.0 - mix_lambda) * q[permutation]
            else:
                logits = model(x).view(-1)
                loss = loss_fn(logits, y)
            if quality:
                loss = (loss * (0.75 + 0.5 * q)).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(y)
        val_prob, val_y, _ = predict_backbone(model, val_loader, device)
        score = safe_auc(val_y, val_prob)
        history.append({"epoch": epoch, "train_loss": total / max(len(train_frame), 1), "val_auroc": score})
        print(f"[CNN] seed={seed} size={size} epoch={epoch}/{EPOCHS} val_auroc={score:.4f}")
        if score > best_score + CNN_MIN_DELTA:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
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
            "rng_state": capture_rng_state(),
        }
        temporary_checkpoint = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary_checkpoint)
        temporary_checkpoint.replace(checkpoint_path)
        if epoch >= CNN_MIN_EPOCHS and stale >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    history_frame = pd.DataFrame(history)
    history_frame["selected_best_epoch"] = int(best_epoch)
    history_frame["selected_best_val_auroc"] = float(best_score)
    return model, history_frame, device


def predict_backbone(model, loader, device):
    import torch

    model.eval()
    probabilities, labels, ids = [], [], []
    with torch.no_grad():
        for x, y, image_ids, _ in loader:
            logits = model(x.to(device)).view(-1)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(y.numpy().astype(int).tolist())
            ids.extend(list(image_ids))
    return np.asarray(probabilities), np.asarray(labels), ids


def predict_backbone_frame(model, frame: pd.DataFrame, size: int, quality: bool, device):
    from torch.utils.data import DataLoader

    loader = DataLoader(
        ImageDataset(frame, size, "none", False, quality).dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    return predict_backbone(model, loader, device)


def extract_features(model, frame: pd.DataFrame, size: int, quality: bool, device, backbone: str = "resnet50"):
    import torch
    from torch.utils.data import DataLoader

    class FeatureDataset(ImageDataset(frame, size, "none", False, quality).dataset.__class__):
        pass

    loader = DataLoader(ImageDataset(frame, size, "none", False, quality).dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    if backbone.lower() in {"efficientnet_b0", "efficientnet-b0", "efficientnetb0"}:
        class EfficientNetFeatures(torch.nn.Module):
            def __init__(self, network):
                super().__init__()
                self.features = network.features
                self.pool = network.avgpool

            def forward(self, values):
                return self.pool(self.features(values))

        feature_model = EfficientNetFeatures(model).to(device)
    else:
        feature_model = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)
    feature_model.eval()
    features, labels, ids, quality_values = [], [], [], []
    with torch.no_grad():
        for x, y, image_ids, q in loader:
            values = feature_model(x.to(device)).flatten(1).cpu().numpy()
            features.append(values)
            labels.extend(y.numpy().astype(int).tolist())
            ids.extend(list(image_ids))
            quality_values.extend(q.numpy().astype(float).tolist())
    values = np.concatenate(features, axis=0) if features else np.empty((0, 2048), dtype=np.float32)
    if quality:
        values = np.column_stack([values, np.asarray(quality_values)])
    return values, np.asarray(labels), ids, np.asarray(quality_values)


def safe_auc(y_true, probabilities):
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.size == 0 or np.unique(y_true).size != 2:
        return np.nan
    return float(roc_auc_score(y_true, probabilities))


def thresholds(y_true, probabilities) -> dict[str, float]:
    y_true, probabilities = np.asarray(y_true).astype(int), np.asarray(probabilities).astype(float)
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    rows = []
    for threshold in candidates:
        pred = (probabilities >= threshold).astype(int)
        tn = int(((y_true == 0) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        tp = int(((y_true == 1) & (pred == 1)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        rows.append((float(threshold), sens, spec, sens + spec - 1))
    table = pd.DataFrame(rows, columns=["threshold", "sensitivity", "specificity", "youden"])
    best = table.sort_values(["youden", "threshold"], ascending=[False, True]).iloc[0]
    result = {"youden": float(best["threshold"])}
    for target, name in [(0.90, "sens90"), (0.95, "sens95")]:
        eligible = table[table["sensitivity"] >= target]
        result[name] = float(eligible.sort_values(["specificity", "threshold"], ascending=[False, True]).iloc[0]["threshold"]) if not eligible.empty else float(table.iloc[0]["threshold"])
    eligible = table[table["specificity"] >= 0.90]
    result["spec90"] = float(eligible.sort_values(["sensitivity", "threshold"], ascending=[False, True]).iloc[0]["threshold"]) if not eligible.empty else float(table.iloc[-1]["threshold"])
    return result


def expected_calibration_error(y_true, probabilities, bins: int = 10) -> float:
    y_true, probabilities = np.asarray(y_true).astype(int), np.asarray(probabilities).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        upper_mask = probabilities <= edges[index + 1] if index == bins - 1 else probabilities < edges[index + 1]
        mask = (probabilities >= edges[index]) & upper_mask
        if mask.any():
            value += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(probabilities[mask].mean()))
    return float(value)


def metrics(y_true, probabilities, threshold, route, seed, split, ratio, size, quality, strategy):
    from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, matthews_corrcoef

    y_true, probabilities = np.asarray(y_true).astype(int), np.asarray(probabilities).astype(float)
    prediction = (probabilities >= threshold).astype(int)
    tn = int(((y_true == 0) & (prediction == 0)).sum())
    fp = int(((y_true == 0) & (prediction == 1)).sum())
    fn = int(((y_true == 1) & (prediction == 0)).sum())
    tp = int(((y_true == 1) & (prediction == 1)).sum())
    return {
        "route": route, "seed": seed, "split": split, "ratio": ratio, "input_size": size, "lidsentry": quality, "class_strategy": strategy,
        "threshold": float(threshold), "accuracy": float(accuracy_score(y_true, prediction)), "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, prediction)), "auroc": safe_auc(y_true, probabilities), "auprc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)), "ece": expected_calibration_error(y_true, probabilities), "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
        "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1), "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def fit_xgb(train_x, train_y, val_x, val_y, seed):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError("xgboost is required for the legacy reproduction") from exc
    params = {"learning_rate": 0.005, "n_estimators": 2000, "early_stopping_rounds": 75, "max_depth": 4, "min_child_weight": 5, "gamma": 0.6, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1, "objective": "binary:logistic", "eval_metric": "logloss", "random_state": seed, "n_jobs": 4}
    if device_name() == "cuda":
        params.update({"tree_method": "hist", "device": "cuda"})
    else:
        params.update({"tree_method": "hist"})
    try:
        model = XGBClassifier(**params)
        model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
        model._sycnn_device = "cuda" if params.get("device") == "cuda" else "cpu"
    except Exception as exc:
        print(f"[WARN] GPU XGBoost failed, retrying CPU: {exc}")
        params.pop("device", None)
        model = XGBClassifier(**params)
        model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
        model._sycnn_device = "cpu_fallback"
    model._sycnn_best_iteration = getattr(model, "best_iteration", None)
    model._sycnn_best_score = getattr(model, "best_score", None)
    return model


def fit_random_forest(train_x, train_y, seed):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        class_weight=None,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    return model


def save_similarity(frame: pd.DataFrame, synthetic: pd.DataFrame, output: Path, seed: int):
    if synthetic.empty:
        return
    route_output = output
    route_output.mkdir(parents=True, exist_ok=True)
    unique_synthetic = synthetic.drop_duplicates("absolute_path").copy()
    sim_max = int(os.environ.get("SIM_MAX", 200))
    synthetic_hashes = [
        {"path": str(path), "md5": md5(Path(path))}
        for path in sorted(unique_synthetic["absolute_path"].astype(str))
    ]
    gan_marker_path = Path(synthetic_hashes[0]["path"]).parent / "gan_complete.json"
    gan_fingerprint = None
    if gan_marker_path.is_file():
        gan_fingerprint = json.loads(gan_marker_path.read_text(encoding="utf-8")).get("configuration_fingerprint")
    cache_payload = {
        "synthetic_hashes": synthetic_hashes,
        "real_md5": sorted(frame["md5"].astype(str)),
        "gan_fingerprint": gan_fingerprint,
        "sim_max": sim_max,
        "algorithm": "all_md5_dhash_plus_class_stratified_ssim224_handcrafted_embedding_v3",
    }
    cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()
    cache_output = OUT / "similarity_cache" / cache_key
    cache_files = [
        cache_output / "synthetic_real_nearest_neighbors.csv",
        cache_output / "synthetic_real_duplicate_flags.csv",
        cache_output / "synthetic_synthetic_duplicate_flags.csv",
        cache_output / "synthetic_real_nearest_neighbor_panels.png",
    ]
    cache_marker = cache_output / "similarity_complete.json"
    if cache_marker.is_file() and all(path.is_file() and path.stat().st_size > 0 for path in cache_files):
        for path in cache_files:
            shutil.copy2(path, route_output / path.name)
        return
    cache_output.mkdir(parents=True, exist_ok=True)
    output = cache_output
    real = frame.copy()
    source_flags = real.assign(_is_original_test=real["original_split"].astype(str).str.lower().eq("test"))
    source_groups = set(source_flags.loc[source_flags["_is_original_test"], "patient_group"])
    real["audit_split"] = real["split"]
    real.loc[real["patient_group"].isin(source_groups), "audit_split"] = "source_holdout"
    real_arrays = []
    for _, row in real.iterrows():
        with Image.open(row["absolute_path"]).convert("L") as image:
            real_arrays.append(np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0)
    def embedding(value):
        compact = value.reshape(16, 4, 16, 4).mean(axis=(1, 3))
        gx = np.diff(compact, axis=1, prepend=compact[:, :1])
        gy = np.diff(compact, axis=0, prepend=compact[:1, :])
        vector = np.concatenate([compact.reshape(-1), gx.reshape(-1), gy.reshape(-1)]).astype(np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-8)

    real_embeddings = [embedding(real_array) for real_array in real_arrays]
    real_dhash_values = np.asarray([int(str(value), 16) for value in real["dhash"]], dtype=object)
    real_md5_to_ids = real.groupby("md5")["image_id"].apply(list).to_dict()
    duplicate_rows = []
    synthetic_self_rows = []
    for _, row in unique_synthetic.iterrows():
        synthetic_md5 = md5(Path(row["absolute_path"]))
        synthetic_dhash = dhash(Path(row["absolute_path"]))
        synthetic_dhash_value = int(synthetic_dhash, 16)
        hamming = np.asarray([(synthetic_dhash_value ^ int(value)).bit_count() for value in real_dhash_values])
        nearest_index = int(np.argmin(hamming))
        duplicate_rows.append({
            "seed": seed,
            "synthetic_id": row["image_id"],
            "synthetic_path": row["absolute_path"],
            "synthetic_md5": synthetic_md5,
            "synthetic_dhash": synthetic_dhash,
            "exact_md5_match": synthetic_md5 in real_md5_to_ids,
            "exact_md5_real_ids": ";".join(real_md5_to_ids.get(synthetic_md5, [])),
            "min_dhash_hamming": int(hamming[nearest_index]),
            "perceptual_near_duplicate_hamming_le_2": int(hamming[nearest_index]) <= 2,
            "nearest_dhash_real_id": real.iloc[nearest_index]["image_id"],
            "nearest_dhash_real_split": real.iloc[nearest_index]["audit_split"],
        })
        synthetic_self_rows.append({
            "synthetic_id": row["image_id"],
            "synthetic_path": row["absolute_path"],
            "synthetic_md5": synthetic_md5,
            "synthetic_dhash": synthetic_dhash,
        })
    pd.DataFrame(duplicate_rows).to_csv(output / "synthetic_real_duplicate_flags.csv", index=False)
    synthetic_self = pd.DataFrame(synthetic_self_rows)
    synthetic_self["exact_md5_duplicate"] = synthetic_self.duplicated("synthetic_md5", keep=False)
    synthetic_self["exact_dhash_duplicate"] = synthetic_self.duplicated("synthetic_dhash", keep=False)
    synthetic_self.to_csv(output / "synthetic_synthetic_duplicate_flags.csv", index=False)
    sampled_parts = []
    for _, class_block in unique_synthetic.groupby("label", sort=True):
        sampled_parts.append(
            class_block.sample(
                n=min(len(class_block), max(sim_max // 2, 1)),
                random_state=seed,
            )
        )
    similarity_sample = pd.concat(sampled_parts, ignore_index=True).head(sim_max)
    syn_rows = []
    for _, row in similarity_sample.iterrows():
        with Image.open(row["absolute_path"]).convert("L") as image:
            array = np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
        array_embedding = embedding(array)
        distances = np.asarray([float(np.linalg.norm(array_embedding - real_embedding)) for real_embedding in real_embeddings])
        index = int(np.argmin(distances))
        try:
            from skimage.metrics import structural_similarity

            with Image.open(row["absolute_path"]).convert("L") as image:
                synthetic_ssim = np.asarray(letterbox_square(image.convert("RGB"), 224).convert("L"), dtype=np.float32) / 255.0
            with Image.open(real.iloc[index]["absolute_path"]).convert("L") as image:
                real_ssim = np.asarray(letterbox_square(image.convert("RGB"), 224).convert("L"), dtype=np.float32) / 255.0
            ssim = float(structural_similarity(synthetic_ssim, real_ssim, data_range=1.0))
            ssim_method = "skimage_structural_similarity"
        except Exception:
            ssim = float(np.clip(1.0 - np.mean((array - real_arrays[index]) ** 2) / max(np.var(array) + np.var(real_arrays[index]) + 1e-6, 1e-6), 0.0, 1.0))
            ssim_method = "fallback_normalized_mse_proxy"
        syn_rows.append({"seed": seed, "synthetic_id": row["image_id"], "synthetic_label": row["label"], "synthetic_path": row["absolute_path"], "nearest_real_id": real.iloc[index]["image_id"], "nearest_real_path": real.iloc[index]["absolute_path"], "nearest_real_split": real.iloc[index]["audit_split"], "nearest_real_original_split": real.iloc[index]["original_split"], "ssim": ssim, "ssim_method": ssim_method, "embedding_distance": float(distances[index]), "embedding_method": "normalized_16x16_grayscale_plus_first_gradients", "neighbor_selection": "minimum_handcrafted_embedding_distance", "n_unique_synthetic_total": int(len(unique_synthetic)), "n_similarity_sampled": int(len(similarity_sample)), "sampling_design": "class_stratified_seeded"})
    similarity_frame = pd.DataFrame(syn_rows)
    similarity_frame.to_csv(output / "synthetic_real_nearest_neighbors.csv", index=False)
    # Keep a compact, publication-ready visual audit for the supplementary material.
    try:
        from PIL import ImageDraw

        panel_rows = []
        panel_selection = pd.concat([
            block.head(6) for _, block in similarity_frame.groupby("synthetic_label", sort=True)
        ], ignore_index=True)
        for _, item in panel_selection.iterrows():
            with Image.open(item["synthetic_path"]).convert("RGB") as synthetic_image:
                syn_image = synthetic_image.resize((192, 192), Image.Resampling.LANCZOS).copy()
            with Image.open(item["nearest_real_path"]).convert("RGB") as real_image:
                real_image = real_image.resize((192, 192), Image.Resampling.LANCZOS).copy()
            canvas = Image.new("RGB", (400, 230), "white")
            canvas.paste(syn_image, (4, 4))
            canvas.paste(real_image, (204, 4))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 201), f"synthetic: {item['synthetic_id']}", fill="black")
            draw.text((206, 201), f"nearest real: {item['nearest_real_id']}", fill="black")
            draw.text((6, 216), f"SSIM={float(item['ssim']):.3f}  dist={float(item['embedding_distance']):.3f}", fill="black")
            panel_rows.append(canvas)
        if panel_rows:
            panel = Image.new("RGB", (400, 230 * len(panel_rows)), "white")
            for index, row_image in enumerate(panel_rows):
                panel.paste(row_image, (0, 230 * index))
            panel.save(output / "synthetic_real_nearest_neighbor_panels.png")
    except Exception as exc:
        print(f"[WARN] nearest-neighbor panel generation failed: {exc}")
    if not all(path.is_file() and path.stat().st_size > 0 for path in cache_files):
        raise RuntimeError(f"Synthetic similarity audit did not produce every required artifact in {cache_output}")
    cache_marker.write_text(json.dumps({
        "status": "complete",
        "cache_key": cache_key,
        "n_unique_synthetic": int(len(unique_synthetic)),
        "n_similarity_sampled": int(len(similarity_sample)),
        "sampling_design": "class_stratified_seeded",
    }, indent=2), encoding="utf-8")
    for path in cache_files:
        shutil.copy2(path, route_output / path.name)


def save_near_duplicate_audit(frame: pd.DataFrame, output: Path):
    audit_payload = frame[["image_id", "split", "md5", "dhash"]].sort_values("image_id").to_csv(index=False)
    audit_fingerprint = hashlib.sha256(audit_payload.encode("utf-8")).hexdigest()
    audit_path = output / "near_duplicate_audit.csv"
    marker_path = output / "near_duplicate_audit_complete.json"
    if audit_path.is_file() and audit_path.stat().st_size > 0 and marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("configuration_fingerprint") == audit_fingerprint:
                return
        except Exception:
            pass
    rows = []
    for column in ["md5", "dhash"]:
        duplicates = frame[frame.duplicated(column, keep=False)].sort_values(column)
        for key, group in duplicates.groupby(column):
            splits = sorted(group["split"].unique())
            rows.append({"key_type": column, "key": key, "n_images": len(group), "splits": ";".join(splits), "cross_split": len(splits) > 1, "image_ids": ";".join(group["image_id"].tolist())})
    dhashes = frame["dhash"].fillna("").astype(str).tolist()
    image_ids = frame["image_id"].tolist()
    for left_index in range(len(dhashes)):
        if not dhashes[left_index]:
            continue
        left_value = int(dhashes[left_index], 16)
        for right_index in range(left_index + 1, len(dhashes)):
            if not dhashes[right_index]:
                continue
            hamming = (left_value ^ int(dhashes[right_index], 16)).bit_count()
            if hamming <= 2:
                splits = sorted({str(frame.iloc[left_index]["split"]), str(frame.iloc[right_index]["split"])})
                rows.append({"key_type": "dhash_hamming_le_2", "key": str(hamming), "n_images": 2, "splits": ";".join(splits), "cross_split": len(splits) > 1, "image_ids": f"{image_ids[left_index]};{image_ids[right_index]}"})
    audit_frame = pd.DataFrame(rows)
    audit_frame.to_csv(audit_path, index=False)
    marker_path.write_text(json.dumps({
        "status": "complete",
        "configuration_fingerprint": audit_fingerprint,
        "n_images": int(len(frame)),
        "cross_split_pairs_or_groups": int(audit_frame.get("cross_split", pd.Series(dtype=bool)).sum()),
    }, indent=2), encoding="utf-8")


def save_calibration_and_decision_curve(prediction_files: list[Path], output: Path):
    calibration_rows, decision_rows = [], []
    decision_thresholds = np.arange(0.05, 0.96, 0.05)
    for path in prediction_files:
        pred = pd.read_csv(path)
        for (route, seed, split), group in pred.groupby(["route", "seed", "split"]):
            if split == "val":
                continue
            y = group["label"].to_numpy(dtype=int)
            prob = group["prob_malignant"].to_numpy(dtype=float)
            bins = np.linspace(0.0, 1.0, 11)
            for index in range(10):
                mask = (prob >= bins[index]) & ((prob < bins[index + 1]) if index < 9 else (prob <= bins[index + 1]))
                if not mask.any():
                    continue
                calibration_rows.append({
                    "route": route, "seed": seed, "split": split, "bin": index,
                    "n": int(mask.sum()), "mean_predicted_risk": float(prob[mask].mean()),
                    "observed_malignant_rate": float(y[mask].mean()),
                })
            prevalence = float(y.mean())
            n = max(len(y), 1)
            for threshold in decision_thresholds:
                predicted = prob >= threshold
                tp = int(np.sum(predicted & (y == 1)))
                fp = int(np.sum(predicted & (y == 0)))
                model_nb = tp / n - fp / n * (threshold / (1.0 - threshold))
                treat_all_nb = prevalence - (1.0 - prevalence) * (threshold / (1.0 - threshold))
                decision_rows.append({
                    "route": route, "seed": seed, "split": split,
                    "threshold": float(threshold), "prevalence": prevalence,
                    "model_net_benefit": float(model_nb),
                    "treat_all_net_benefit": float(treat_all_nb),
                    "treat_none_net_benefit": 0.0,
                    "tp": tp, "fp": fp, "n": n,
                })
    pd.DataFrame(calibration_rows).to_csv(output / "calibration_bins.csv", index=False)
    pd.DataFrame(decision_rows).to_csv(output / "decision_curve_net_benefit.csv", index=False)


def run_route(frame: pd.DataFrame, split_audit: dict, seed: int, ratio: float, size: int, quality: bool, strategy: str, mode: str, pool_dir: Path | None, source_holdout: bool = False, backbone: str = "resnet50", label_noise: float = 0.0):
    started_at = datetime.now(timezone.utc).isoformat()
    if source_holdout:
        source_flags = frame.assign(_is_original_test=frame["original_split"].astype(str).str.lower().eq("test"))
        source_groups = set(source_flags.loc[source_flags["_is_original_test"], "patient_group"])
        source_frame = frame[frame["patient_group"].isin(source_groups)].copy()
        analysis_frame = frame[~frame["patient_group"].isin(source_groups)].copy()
        train = analysis_frame[analysis_frame["split"] == "train"].copy()
        val = analysis_frame[analysis_frame["split"] == "val"].copy()
        test1 = analysis_frame[analysis_frame["split"] == "test1"].copy()
        test2 = source_frame
        split_label = "source_holdout"
    else:
        train = frame[frame["split"] == "train"].copy()
        val = frame[frame["split"] == "val"].copy()
        test1 = frame[frame["split"] == "test1"].copy()
        test2 = frame[frame["split"] == "test2"].copy()
        split_label = "test2"
    train["label_before_noise"] = train["label"].astype(int)
    if label_noise > 0:
        noise_rng = np.random.default_rng(seed + int(label_noise * 10000) + 7919)
        noisy_indices = noise_rng.choice(train.index.to_numpy(), size=int(round(len(train) * label_noise)), replace=False)
        train.loc[noisy_indices, "label"] = 1 - train.loc[noisy_indices, "label"].astype(int)
    train["label_was_flipped"] = train["label"].astype(int).ne(train["label_before_noise"])
    synthetic = synthetic_frame(pool_dir, train, ratio, strategy, seed) if pool_dir is not None else pd.DataFrame()
    train_for_model = pd.concat([train.assign(synthetic=False), synthetic], ignore_index=True)
    route = f"{mode}_n{ratio:g}_{strategy}_{'lidsentry' if quality else 'nolidsentry'}_noise{label_noise:g}_size{size}"
    run_dir = OUT / "runs" / f"seed_{seed}" / route
    run_dir.mkdir(parents=True, exist_ok=True)
    is_strong_baseline = str(mode).startswith("strong_baseline")
    run_head_ablation = (
        mode == "primary_gan_resnet50_xgb"
        and strategy == "original_ratio"
        and float(ratio) in {0.0, 4.0}
        and float(label_noise) == 0.0
        and not source_holdout
    ) or is_strong_baseline
    split_source = Path(split_audit["split_source"])
    route_configuration = {
        "route": route,
        "seed": seed,
        "ratio": float(ratio),
        "input_size": int(size),
        "backbone": backbone,
        "lidsentry": bool(quality),
        "class_strategy": strategy,
        "class_strategy_definition": (
            "synthetic class counts match real training prevalence"
            if strategy == "original_ratio" else
            "synthetic counts chosen so the final real-plus-synthetic training set is exactly 1:1"
        ),
        "label_noise_rate": float(label_noise),
        "strong_baseline": is_strong_baseline,
        "source_holdout": bool(source_holdout),
        "backbone_initialization": BACKBONE_INIT,
        "backbone_weights": "ResNet50_Weights.IMAGENET1K_V2" if backbone.lower() == "resnet50" else "ImageNet torchvision weights",
        "pretrained_weights_used": BACKBONE_INIT == "imagenet",
        "downstream_classifier": "XGBoost",
        "training_augmentation": (
            ["letterbox224", "horizontal_flip", "rotation8", "color_jitter", "random_affine", "mixup"]
            if is_strong_baseline else
            ["letterbox224", "horizontal_flip", "rotation8"] + (["quality_conditioned_degradation"] if quality else [])
        ),
        "class_weighted_loss": bool(is_strong_baseline),
        "mixup_alpha": 0.4 if is_strong_baseline else None,
        "quality_aware_components": (
            ["quality_conditioned_degradation", "quality_weighted_backbone_loss", "quality_scalar_xgboost_feature"] if quality else []
        ),
        "synthetic_sampling_with_replacement": False,
        "gan_pool_per_class": int(os.environ.get("GAN_POOL_PER_CLASS", "2048")),
        "gan_native_image_size": GAN_SIZE,
        "epochs": EPOCHS,
        "gan_epochs": GAN_EPOCHS,
        "gan_batch_size": GAN_BATCH_SIZE,
        "gan_patience": GAN_PATIENCE,
        "gan_patience_interval_epochs": GAN_PATIENCE_INTERVAL_EPOCHS,
        "gan_min_epochs": GAN_MIN_EPOCHS,
        "gan_min_delta": GAN_MIN_DELTA,
        "gan_structure_first": GAN_STRUCTURE_FIRST,
        "gan_objective": GAN_OBJECTIVE,
        "gan_spectral_norm": GAN_SPECTRAL_NORM,
        "gan_discriminator_multiscale": GAN_DISCRIMINATOR_MULTISCALE,
        "gan_discriminator_patch_blocks": GAN_DISCRIMINATOR_PATCH_BLOCKS,
        "gan_feature_match_weight": GAN_FEATURE_MATCH_WEIGHT,
        "gan_discriminator_augment": GAN_DISCRIMINATOR_AUGMENT,
        "gan_discriminator_augment_prob": GAN_DISCRIMINATOR_AUGMENT_PROB,
        "gan_discriminator_augment_translation": GAN_DISCRIMINATOR_AUGMENT_TRANSLATION,
        "gan_early_stopping_enabled": bool(GAN_MIN_EPOCHS < GAN_EPOCHS or GAN_PATIENCE < GAN_EPOCHS),
        "patience": PATIENCE,
        "cnn_min_epochs": CNN_MIN_EPOCHS,
        "cnn_min_delta": CNN_MIN_DELTA,
        "classifier_head_ablation": bool(run_head_ablation),
        "random_forest_parameters": ({
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
            "class_weight": None,
        } if run_head_ablation else None),
        "batch_size": BATCH_SIZE,
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "locked_split_sha256": sha256_file(split_source),
    }
    configuration_fingerprint = hashlib.sha256(
        json.dumps(route_configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()
    required_outputs = [
        run_dir / "training_manifest.csv",
        run_dir / "metrics_operating_points.csv",
        run_dir / "predictions.csv",
        run_dir / "thresholds.json",
        run_dir / "history.csv",
        run_dir / "training_diagnostics.json",
        run_dir / "backbone_best_state.pt",
        run_dir / "xgboost_model.ubj",
        run_dir / "run_manifest.json",
        run_dir / "run_complete.json",
    ]
    if ratio > 0:
        required_outputs.extend([
            run_dir / "similarity" / "synthetic_real_nearest_neighbors.csv",
            run_dir / "similarity" / "synthetic_real_duplicate_flags.csv",
            run_dir / "similarity" / "synthetic_synthetic_duplicate_flags.csv",
            run_dir / "similarity" / "synthetic_real_nearest_neighbor_panels.png",
        ])
    if run_head_ablation:
        required_outputs.extend([
            run_dir / "classifier_head_ablation" / "metrics_operating_points.csv",
            run_dir / "classifier_head_ablation" / "predictions.csv",
            run_dir / "classifier_head_ablation" / "thresholds.json",
            run_dir / "classifier_head_ablation" / "random_forest_model.joblib",
            run_dir / "classifier_head_ablation" / "ablation_complete.json",
        ])
    can_skip = not FORCE_RERUN and all(path.is_file() and path.stat().st_size > 0 for path in required_outputs)
    if can_skip:
        try:
            previous_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            complete_marker = json.loads((run_dir / "run_complete.json").read_text(encoding="utf-8"))
            can_skip = (
                previous_manifest.get("configuration_fingerprint") == configuration_fingerprint
                and complete_marker.get("configuration_fingerprint") == configuration_fingerprint
                and complete_marker.get("status") == "complete"
            )
        except Exception:
            can_skip = False
    if can_skip:
        print(f"[SKIP] complete route already exists: {route}")
        return run_dir
    training_manifest = train_for_model.copy()
    training_manifest["generation_source"] = np.where(training_manifest["synthetic"].astype(bool), "GAN_train_only", "real_train_only")
    training_manifest["synthetic_ratio"] = float(ratio)
    training_manifest["real_training_images"] = int(len(train))
    training_manifest["synthetic_training_images"] = int(len(synthetic))
    training_manifest["realized_synthetic_to_real_ratio"] = float(len(synthetic) / max(len(train), 1))
    final_counts = train_for_model["label"].value_counts().to_dict()
    training_manifest["final_benign_count"] = int(final_counts.get(0, 0))
    training_manifest["final_malignant_count"] = int(final_counts.get(1, 0))
    training_manifest["final_class_ratio_benign_to_malignant"] = float(final_counts.get(0, 0) / max(final_counts.get(1, 0), 1))
    training_manifest["class_generation_strategy"] = strategy
    training_manifest["label_noise_rate"] = float(label_noise)
    training_manifest.to_csv(run_dir / "training_manifest.csv", index=False)
    model, history, device = train_backbone(
        train_for_model,
        val,
        size,
        "strong" if is_strong_baseline else ("degradation" if quality else "legacy"),
        quality,
        is_strong_baseline,
        seed,
        backbone,
        run_dir,
        configuration_fingerprint,
    )
    import torch

    final_backbone_path = run_dir / "backbone_best_state.pt"
    temporary_backbone_path = run_dir / "backbone_best_state.tmp"
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "backbone": backbone,
        "configuration_fingerprint": configuration_fingerprint,
    }, temporary_backbone_path)
    temporary_backbone_path.replace(final_backbone_path)
    cnn_val_prob, cnn_val_y, cnn_val_ids = predict_backbone_frame(model, val, size, quality, device)
    cnn_test1_prob, cnn_test1_y, cnn_test1_ids = predict_backbone_frame(model, test1, size, quality, device)
    cnn_test2_prob, cnn_test2_y, cnn_test2_ids = (
        predict_backbone_frame(model, test2, size, quality, device) if not test2.empty else (None, None, [])
    )
    train_x, train_y, train_ids, _ = extract_features(model, train_for_model, size, quality, device, backbone)
    val_x, val_y, val_ids, _ = extract_features(model, val, size, quality, device, backbone)
    test1_x, test1_y, test1_ids, _ = extract_features(model, test1, size, quality, device, backbone)
    test2_x, test2_y, test2_ids, _ = extract_features(model, test2, size, quality, device, backbone) if not test2.empty else (None, None, [], None)
    if not (
        np.array_equal(cnn_val_y, val_y)
        and np.array_equal(cnn_test1_y, test1_y)
        and cnn_val_ids == val_ids
        and cnn_test1_ids == test1_ids
    ):
        raise RuntimeError("Direct CNN predictions are not aligned with the locked split")
    classifier = fit_xgb(train_x, train_y, val_x, val_y, seed)
    temporary_xgb_path = run_dir / "xgboost_model.tmp.ubj"
    classifier.save_model(str(temporary_xgb_path))
    temporary_xgb_path.replace(run_dir / "xgboost_model.ubj")
    val_prob = classifier.predict_proba(val_x)[:, 1]
    test1_prob = classifier.predict_proba(test1_x)[:, 1]
    test2_prob = classifier.predict_proba(test2_x)[:, 1] if test2_x is not None else None
    selected = thresholds(val_y, val_prob)
    (run_dir / "thresholds.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    all_metrics, prediction_rows = [], []
    evaluations = [("val", val_y, val_prob, val_ids), ("test1", test1_y, test1_prob, test1_ids)]
    if test2_prob is not None:
        evaluations.append((split_label, test2_y, test2_prob, test2_ids))
    for split_name, y_true, prob, ids in evaluations:
        prediction_rows.append(pd.DataFrame({"route": route, "seed": seed, "split": split_name, "image_id": ids, "label": y_true, "prob_malignant": prob, "patient_group": frame.set_index("image_id").reindex(ids)["patient_group"].fillna("synthetic").to_numpy()}))
        for point, threshold in selected.items():
            all_metrics.append({**metrics(y_true, prob, threshold, route, seed, split_name, ratio, size, quality, strategy), "operating_point": point})
    pd.DataFrame(all_metrics).to_csv(run_dir / "metrics_operating_points.csv", index=False)
    xgb_prediction_frame = pd.concat(prediction_rows, ignore_index=True)
    xgb_prediction_frame.to_csv(run_dir / "predictions.csv", index=False)
    if run_head_ablation:
        import joblib

        ablation_dir = run_dir / "classifier_head_ablation"
        ablation_dir.mkdir(parents=True, exist_ok=True)
        random_forest = fit_random_forest(train_x, train_y, seed)
        temporary_rf_path = ablation_dir / "random_forest_model.tmp.joblib"
        joblib.dump(random_forest, temporary_rf_path)
        temporary_rf_path.replace(ablation_dir / "random_forest_model.joblib")
        rf_val_prob = random_forest.predict_proba(val_x)[:, 1]
        rf_test1_prob = random_forest.predict_proba(test1_x)[:, 1]
        rf_test2_prob = random_forest.predict_proba(test2_x)[:, 1] if test2_x is not None else None
        head_probabilities = {
            "cnn": {"val": cnn_val_prob, "test1": cnn_test1_prob, split_label: cnn_test2_prob},
            "xgboost": {"val": val_prob, "test1": test1_prob, split_label: test2_prob},
            "random_forest": {"val": rf_val_prob, "test1": rf_test1_prob, split_label: rf_test2_prob},
        }
        head_thresholds = {
            "cnn": thresholds(cnn_val_y, cnn_val_prob),
            "xgboost": selected,
            "random_forest": thresholds(val_y, rf_val_prob),
        }
        head_metrics, head_predictions = [], []
        evaluation_ids = {"val": val_ids, "test1": test1_ids, split_label: test2_ids}
        evaluation_labels = {"val": val_y, "test1": test1_y, split_label: test2_y}
        for classifier_head, probabilities_by_split in head_probabilities.items():
            head_route = f"{route}__head_{classifier_head}"
            for current_split, probabilities in probabilities_by_split.items():
                if probabilities is None:
                    continue
                ids = evaluation_ids[current_split]
                labels = evaluation_labels[current_split]
                head_predictions.append(pd.DataFrame({
                    "route": head_route,
                    "backbone_route": route,
                    "classifier_head": classifier_head,
                    "seed": seed,
                    "split": current_split,
                    "image_id": ids,
                    "label": labels,
                    "prob_malignant": probabilities,
                    "patient_group": frame.set_index("image_id").reindex(ids)["patient_group"].fillna("synthetic").to_numpy(),
                }))
                for point, threshold in head_thresholds[classifier_head].items():
                    row = metrics(labels, probabilities, threshold, head_route, seed, current_split, ratio, size, quality, strategy)
                    row.update({
                        "operating_point": point,
                        "backbone_route": route,
                        "classifier_head": classifier_head,
                        "threshold_source": "validation_locked",
                    })
                    head_metrics.append(row)
        pd.DataFrame(head_metrics).to_csv(ablation_dir / "metrics_operating_points.csv", index=False)
        pd.concat(head_predictions, ignore_index=True).to_csv(ablation_dir / "predictions.csv", index=False)
        (ablation_dir / "thresholds.json").write_text(json.dumps(head_thresholds, indent=2), encoding="utf-8")
        (ablation_dir / "ablation_complete.json").write_text(json.dumps({
            "status": "complete",
            "backbone_route": route,
            "seed": seed,
            "classifier_heads": ["cnn", "xgboost", "random_forest"],
            "shared_resnet_features": True,
            "configuration_fingerprint": configuration_fingerprint,
        }, indent=2), encoding="utf-8")
    save_similarity(frame, synthetic, run_dir / "similarity", seed)
    history.to_csv(run_dir / "history.csv", index=False)
    completed_epochs = int(history["epoch"].max()) if not history.empty else 0
    best_epoch = int(history.iloc[0]["selected_best_epoch"]) if not history.empty else 0
    best_val_auroc = float(history.iloc[0]["selected_best_val_auroc"]) if not history.empty else np.nan
    final_val_auroc = float(history.iloc[-1]["val_auroc"]) if not history.empty else np.nan
    tail = history.tail(min(10, len(history)))
    tail_trend = float(np.polyfit(tail["epoch"], tail["val_auroc"], 1)[0]) if len(tail) >= 2 else 0.0
    diagnostics = {
        "route": route,
        "seed": seed,
        "max_epochs": EPOCHS,
        "minimum_epochs_before_early_stop": CNN_MIN_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "minimum_auroc_improvement": CNN_MIN_DELTA,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "final_epoch_val_auroc": final_val_auroc,
        "best_to_final_auroc_drop": float(best_val_auroc - final_val_auroc),
        "last_10_epoch_auroc_slope": tail_trend,
        "restored_best_validation_state": True,
        "stopped_early": completed_epochs < EPOCHS,
        "reached_minimum_epochs": completed_epochs >= CNN_MIN_EPOCHS,
        "potential_underfit_warning": bool(completed_epochs >= EPOCHS and tail_trend > 2e-4),
        "late_epoch_overfit_warning": bool(best_val_auroc - final_val_auroc > 0.02),
        "warning_interpretation": "diagnostic only; best validation state is restored before XGBoost feature extraction",
    }
    (run_dir / "training_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    torch_config = {
        **route_configuration,
        "configuration_fingerprint": configuration_fingerprint,
        "device_for_training": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
        "xgboost_device": getattr(classifier, "_sycnn_device", "unknown"),
        "xgboost_best_iteration": getattr(classifier, "_sycnn_best_iteration", None),
        "xgboost_best_score": getattr(classifier, "_sycnn_best_score", None),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "package_versions": package_versions(),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_audit": split_audit,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(torch_config, indent=2), encoding="utf-8")
    (run_dir / "run_complete.json").write_text(
        json.dumps({"route": route, "seed": seed, "status": "complete", "configuration_fingerprint": configuration_fingerprint}, indent=2),
        encoding="utf-8",
    )
    checkpoint_path = run_dir / "backbone_checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return run_dir


def bootstrap_ci(prediction_files: list[Path], output: Path, n_boot: int = 1000):
    rows = []
    for path in prediction_files:
        pred = pd.read_csv(path)
        for (route, seed, split), group in pred.groupby(["route", "seed", "split"]):
            if split == "val":
                continue
            rng = np.random.default_rng(int(seed) + 9001)
            patient_groups = [part for _, part in group.groupby("patient_group")]
            threshold_path = path.parent / "thresholds.json"
            selected_thresholds = json.loads(threshold_path.read_text(encoding="utf-8")) if threshold_path.exists() else thresholds(group["label"], group["prob_malignant"])
            values = []
            for _ in range(n_boot):
                sample = pd.concat([patient_groups[index] for index in rng.integers(0, len(patient_groups), len(patient_groups))], ignore_index=True)
                values.append({point: metrics(sample["label"], sample["prob_malignant"], float(selected_thresholds[point]), route, seed, split, 0, 224, "_lidsentry_" in str(route), "unknown") for point in ["youden", "sens90", "sens95", "spec90"]})
            for point in ["youden", "sens90", "sens95", "spec90"]:
                observed = metrics(group["label"], group["prob_malignant"], float(selected_thresholds[point]), route, seed, split, 0, 224, "_lidsentry_" in str(route), "unknown")
                for metric in ["accuracy", "f1", "mcc", "auroc", "auprc", "brier", "sensitivity", "specificity", "fpr", "fnr", "ppv", "npv", "ece"]:
                    values_metric = np.asarray([value[point][metric] for value in values], dtype=float)
                    finite = values_metric[np.isfinite(values_metric)]
                    rows.append({"route": route, "seed": seed, "split": split, "operating_point": point, "metric": metric, "estimate": float(observed[metric]), "bootstrap_mean": float(finite.mean()) if len(finite) else np.nan, "ci95_low": float(np.nanpercentile(finite, 2.5)) if len(finite) else np.nan, "ci95_high": float(np.nanpercentile(finite, 97.5)) if len(finite) else np.nan, "bootstrap_replicates": n_boot, "valid_bootstrap_replicates": int(len(finite)), "resampling_unit": "patient_group", "threshold_source": "validation_locked"})
    pd.DataFrame(rows).to_csv(output / "final_95ci_patient_group_bootstrap.csv", index=False)


def route_metadata(route: str) -> dict | None:
    match = re.match(
        r"^(?P<mode>.+)_n(?P<ratio>[0-9.]+)_(?P<strategy>original_ratio|balanced)_"
        r"(?P<quality>nolidsentry|lidsentry)_noise(?P<noise>[0-9.]+)_size(?P<size>\d+)$",
        str(route),
    )
    if not match:
        return None
    result = match.groupdict()
    result["ratio"] = float(result["ratio"])
    result["noise"] = float(result["noise"])
    result["size"] = int(result["size"])
    result["quality"] = result["quality"] == "lidsentry"
    return result


def is_targeted_pair(left_route: str, right_route: str) -> bool:
    left = route_metadata(left_route)
    right = route_metadata(right_route)
    if left is None or right is None or left["size"] != right["size"]:
        return False
    same = lambda key: left[key] == right[key]
    # Primary comparisons change exactly one experimental factor.
    if left["mode"] == right["mode"] == "primary_gan_resnet50_xgb":
        if all(same(key) for key in ["ratio", "strategy", "noise"]) and left["quality"] != right["quality"]:
            return True
        if all(same(key) for key in ["ratio", "quality", "noise"]) and left["strategy"] != right["strategy"]:
            return True
        if all(same(key) for key in ["strategy", "quality", "noise"]) and {left["ratio"], right["ratio"]} == {0.0, 4.0}:
            return True
    # Compare the corrected real-only route with the conventional strong baseline.
    modes = {left["mode"], right["mode"]}
    if modes == {"primary_gan_resnet50_xgb", "strong_baseline_resnet50"}:
        return all(
            item["ratio"] == 0.0 and item["strategy"] == "original_ratio"
            and item["quality"] is False and item["noise"] == 0.0
            for item in [left, right]
        )
    # Label-noise sensitivity uses the matched primary n=0 route as noise=0 reference.
    if modes == {"primary_gan_resnet50_xgb", "label_noise_robustness"}:
        primary = left if left["mode"] == "primary_gan_resnet50_xgb" else right
        noisy = right if primary is left else left
        return (
            primary["ratio"] == noisy["ratio"] == 0.0
            and primary["strategy"] == noisy["strategy"] == "original_ratio"
            and primary["quality"] == noisy["quality"]
            and primary["noise"] == 0.0 < noisy["noise"]
        )
    return False


def paired_bootstrap_ci(prediction_files: list[Path], output: Path, n_boot: int = 1000):
    frames = [pd.read_csv(path).assign(_path=str(path)) for path in prediction_files]
    if not frames:
        return
    predictions = pd.concat(frames, ignore_index=True)
    rows = []
    metrics_to_report = ["accuracy", "f1", "mcc", "auroc", "auprc", "sensitivity", "specificity", "fpr", "fnr", "ppv", "npv", "brier", "ece"]
    for (seed, split), block in predictions.groupby(["seed", "split"]):
        if split == "val":
            continue
        route_names = sorted(block["route"].unique())
        for left_index, left_route in enumerate(route_names):
            for right_route in route_names[left_index + 1:]:
                if not is_targeted_pair(left_route, right_route):
                    continue
                left = block[block["route"] == left_route].set_index("image_id")
                right = block[block["route"] == right_route].set_index("image_id")
                common_ids = left.index.intersection(right.index)
                if len(common_ids) == 0:
                    continue
                paired = left.loc[common_ids, ["label", "prob_malignant", "patient_group"]].rename(columns={"label": "label_left", "prob_malignant": "prob_left", "patient_group": "patient_group_left"}).join(
                    right.loc[common_ids, ["label", "prob_malignant", "patient_group"]].rename(columns={"label": "label_right", "prob_malignant": "prob_right", "patient_group": "patient_group_right"}), how="inner"
                )
                if not np.array_equal(paired["label_left"].to_numpy(), paired["label_right"].to_numpy()):
                    continue
                groups = [part for _, part in paired.groupby("patient_group_left")]
                left_path = Path(block[block["route"] == left_route]["_path"].iloc[0])
                right_path = Path(block[block["route"] == right_route]["_path"].iloc[0])
                left_thresholds = json.loads((left_path.parent / "thresholds.json").read_text(encoding="utf-8")) if (left_path.parent / "thresholds.json").exists() else {}
                right_thresholds = json.loads((right_path.parent / "thresholds.json").read_text(encoding="utf-8")) if (right_path.parent / "thresholds.json").exists() else {}
                for point in ["youden", "sens90", "sens95", "spec90"]:
                    threshold_left = float(left_thresholds.get(point, 0.5))
                    threshold_right = float(right_thresholds.get(point, 0.5))
                    pair_key = f"{left_route}|{right_route}|{split}".encode("utf-8")
                    pair_offset = int(hashlib.md5(pair_key).hexdigest()[:8], 16)
                    rng = np.random.default_rng(int(seed) + 17001 + pair_offset)
                    sampled_deltas = {metric: [] for metric in metrics_to_report}
                    for _ in range(n_boot):
                        sample = pd.concat([groups[index] for index in rng.integers(0, len(groups), len(groups))], ignore_index=True)
                        left_metrics = metrics(sample["label_left"], sample["prob_left"], threshold_left, left_route, seed, split, 0, 224, "_lidsentry_" in left_route, "paired")
                        right_metrics = metrics(sample["label_right"], sample["prob_right"], threshold_right, right_route, seed, split, 0, 224, "_lidsentry_" in right_route, "paired")
                        for metric in metrics_to_report:
                            sampled_deltas[metric].append(float(right_metrics[metric] - left_metrics[metric]))
                    for metric, values in sampled_deltas.items():
                        values = np.asarray(values, dtype=float)
                        rows.append({"route_left": left_route, "route_right": right_route, "seed": seed, "split": split, "operating_point": point, "metric": metric, "estimate_delta_right_minus_left": float(values.mean()), "ci95_low": float(np.nanpercentile(values, 2.5)), "ci95_high": float(np.nanpercentile(values, 97.5)), "bootstrap_replicates": n_boot, "resampling_unit": "patient_group", "matched_images": int(len(common_ids))})
    columns = ["route_left", "route_right", "seed", "split", "operating_point", "metric", "estimate_delta_right_minus_left", "ci95_low", "ci95_high", "bootstrap_replicates", "resampling_unit", "matched_images"]
    pd.DataFrame(rows, columns=columns).to_csv(output / "paired_patient_group_bootstrap_deltas.csv", index=False)


def factorial_interaction_bootstrap(prediction_files: list[Path], output: Path, n_boot: int = 1000):
    prediction_map = {}
    threshold_map = {}
    parsed_map = {}
    for path in prediction_files:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        route = str(frame["route"].iloc[0])
        parsed = route_metadata(route)
        if (
            parsed is None
            or parsed["mode"] != "primary_gan_resnet50_xgb"
            or parsed["strategy"] != "original_ratio"
            or parsed["noise"] != 0
            or parsed["size"] != 224
        ):
            continue
        prediction_map[route] = frame[frame["split"].eq("test1")].copy()
        threshold_map[route] = json.loads((path.parent / "thresholds.json").read_text(encoding="utf-8"))
        parsed_map[(parsed["ratio"], parsed["quality"])] = route
    base_routes = [parsed_map.get((0.0, False)), parsed_map.get((0.0, True))]
    if any(route is None for route in base_routes):
        raise RuntimeError("The real-only 2x2 factorial controls are incomplete")
    metric_names = ["auroc", "auprc", "accuracy", "f1", "mcc", "brier", "ece", "sensitivity", "specificity", "fpr", "fnr", "ppv", "npv"]
    lower_is_better = {"brier", "ece", "fpr", "fnr"}
    rows = []
    for ratio in sorted(value for value, quality in parsed_map if value > 0 and quality is False):
        routes = {
            "real_only": parsed_map.get((0.0, False)),
            "quality_only": parsed_map.get((0.0, True)),
            "gan_only": parsed_map.get((ratio, False)),
            "gan_quality": parsed_map.get((ratio, True)),
        }
        if any(route is None for route in routes.values()):
            continue
        merged = None
        for role, route in routes.items():
            block = prediction_map[route][["image_id", "label", "patient_group", "prob_malignant"]].rename(
                columns={"prob_malignant": f"prob_{role}"}
            )
            if merged is None:
                merged = block
            else:
                identity_check = merged[["image_id", "label", "patient_group"]].merge(
                    block[["image_id", "label", "patient_group"]],
                    on="image_id", how="inner", suffixes=("_reference", "_candidate"),
                    validate="one_to_one",
                )
                if (
                    not identity_check["label_reference"].equals(identity_check["label_candidate"])
                    or not identity_check["patient_group_reference"].astype(str).equals(identity_check["patient_group_candidate"].astype(str))
                ):
                    raise RuntimeError(f"Factorial routes do not share labels/groups at n={ratio:g}")
                merged = merged.merge(
                    block.drop(columns=["label", "patient_group"]),
                    on="image_id",
                    how="inner",
                    validate="one_to_one",
                )
        if merged is None or merged.empty:
            continue
        grouped_indices = [
            np.flatnonzero(merged["patient_group"].astype(str).to_numpy() == group)
            for group in sorted(merged["patient_group"].astype(str).unique())
        ]
        rng = np.random.default_rng(int(CFG.get("seed", SEEDS[0])) + int(ratio * 1000) + 2909)
        for point in ["youden", "sens90", "sens95", "spec90"]:
            estimates = {}
            for role, route in routes.items():
                estimates[role] = metrics(
                    merged["label"], merged[f"prob_{role}"], float(threshold_map[route][point]),
                    route, int(CFG.get("seed", SEEDS[0])), "test1", ratio, 224,
                    role in {"quality_only", "gan_quality"}, "original_ratio",
                )
            samples = {name: [] for name in metric_names}
            for _ in range(n_boot):
                selected_groups = rng.integers(0, len(grouped_indices), len(grouped_indices))
                indices = np.concatenate([grouped_indices[index] for index in selected_groups])
                sampled = merged.iloc[indices]
                sampled_metrics = {}
                for role, route in routes.items():
                    sampled_metrics[role] = metrics(
                        sampled["label"], sampled[f"prob_{role}"], float(threshold_map[route][point]),
                        route, int(CFG.get("seed", SEEDS[0])), "test1", ratio, 224,
                        role in {"quality_only", "gan_quality"}, "original_ratio",
                    )
                for name in metric_names:
                    interaction = (
                        sampled_metrics["gan_quality"][name] - sampled_metrics["gan_only"][name]
                        - sampled_metrics["quality_only"][name] + sampled_metrics["real_only"][name]
                    )
                    samples[name].append(float(interaction))
            for name in metric_names:
                interaction = (
                    estimates["gan_quality"][name] - estimates["gan_only"][name]
                    - estimates["quality_only"][name] + estimates["real_only"][name]
                )
                values = np.asarray(samples[name], dtype=float)
                finite = values[np.isfinite(values)]
                orientation = -1.0 if name in lower_is_better else 1.0
                p_value = float(min(1.0, 2 * min(np.mean(finite <= 0), np.mean(finite >= 0)))) if len(finite) else np.nan
                raw_low = float(np.nanpercentile(finite, 2.5)) if len(finite) else np.nan
                raw_high = float(np.nanpercentile(finite, 97.5)) if len(finite) else np.nan
                rows.append({
                    "seed": int(CFG.get("seed", SEEDS[0])), "ratio": ratio, "split": "test1",
                    "operating_point": point, "metric": name,
                    "interaction_raw": float(interaction),
                    "interaction_benefit_oriented": float(interaction * orientation),
                    "ci95_low_raw": raw_low,
                    "ci95_high_raw": raw_high,
                    "ci95_low_benefit_oriented": -raw_high if orientation < 0 else raw_low,
                    "ci95_high_benefit_oriented": -raw_low if orientation < 0 else raw_high,
                    "two_sided_bootstrap_p": p_value,
                    "bootstrap_replicates": n_boot, "resampling_unit": "patient_group",
                    "matched_images": int(len(merged)),
                    "contrast": "(GAN+LidSentry-GAN_only)-(LidSentry_only-real_only)",
                    "threshold_source": "validation_locked",
                    **{f"route_{role}": route for role, route in routes.items()},
                })
    pd.DataFrame(rows).to_csv(output / "factorial_2x2_interaction_bootstrap.csv", index=False)


def classifier_head_ablation_summary(run_dirs: list[Path], output: Path, n_boot: int = 1000):
    metric_frames, prediction_frames, paired_rows = [], [], []
    metrics_to_report = [
        "accuracy", "f1", "mcc", "auroc", "auprc", "brier", "ece",
        "sensitivity", "specificity", "fpr", "fnr", "ppv", "npv",
    ]
    for run_dir in run_dirs:
        ablation_dir = run_dir / "classifier_head_ablation"
        metric_path = ablation_dir / "metrics_operating_points.csv"
        prediction_path = ablation_dir / "predictions.csv"
        threshold_path = ablation_dir / "thresholds.json"
        if not all(path.is_file() and path.stat().st_size > 0 for path in [metric_path, prediction_path, threshold_path]):
            continue
        metric_frames.append(pd.read_csv(metric_path))
        predictions = pd.read_csv(prediction_path)
        prediction_frames.append(predictions)
        test = predictions[predictions["split"].eq("test1")]
        xgb = test[test["classifier_head"].eq("xgboost")].set_index("image_id")
        rf = test[test["classifier_head"].eq("random_forest")].set_index("image_id")
        common_ids = xgb.index.intersection(rf.index)
        if len(common_ids) == 0:
            continue
        paired = xgb.loc[common_ids, ["label", "prob_malignant", "patient_group", "backbone_route"]].rename(
            columns={"prob_malignant": "prob_xgboost"}
        ).join(rf.loc[common_ids, ["label", "prob_malignant", "patient_group"]].rename(
            columns={"label": "label_rf", "prob_malignant": "prob_random_forest", "patient_group": "patient_group_rf"}
        ))
        if not np.array_equal(paired["label"].to_numpy(), paired["label_rf"].to_numpy()):
            raise RuntimeError(f"Classifier-head labels differ in {run_dir.name}")
        if not paired["patient_group"].astype(str).equals(paired["patient_group_rf"].astype(str)):
            raise RuntimeError(f"Classifier-head patient groups differ in {run_dir.name}")
        threshold_values = json.loads(threshold_path.read_text(encoding="utf-8"))
        groups = [part for _, part in paired.groupby("patient_group")]
        seed = int(test["seed"].iloc[0])
        pair_offset = int(hashlib.md5(run_dir.name.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed + 47017 + pair_offset)
        for point in ["youden", "sens90", "sens95", "spec90"]:
            xgb_threshold = float(threshold_values["xgboost"][point])
            rf_threshold = float(threshold_values["random_forest"][point])
            full_xgb = metrics(paired["label"], paired["prob_xgboost"], xgb_threshold, run_dir.name, seed, "test1", 0, 224, False, "head_ablation")
            full_rf = metrics(paired["label"], paired["prob_random_forest"], rf_threshold, run_dir.name, seed, "test1", 0, 224, False, "head_ablation")
            samples = {name: [] for name in metrics_to_report}
            for _ in range(n_boot):
                sample = pd.concat([groups[index] for index in rng.integers(0, len(groups), len(groups))], ignore_index=True)
                sample_xgb = metrics(sample["label"], sample["prob_xgboost"], xgb_threshold, run_dir.name, seed, "test1", 0, 224, False, "head_ablation")
                sample_rf = metrics(sample["label"], sample["prob_random_forest"], rf_threshold, run_dir.name, seed, "test1", 0, 224, False, "head_ablation")
                for name in metrics_to_report:
                    samples[name].append(float(sample_rf[name] - sample_xgb[name]))
            for name, values in samples.items():
                values = np.asarray(values, dtype=float)
                finite = values[np.isfinite(values)]
                paired_rows.append({
                    "backbone_route": str(paired["backbone_route"].iloc[0]),
                    "seed": seed,
                    "split": "test1",
                    "operating_point": point,
                    "metric": name,
                    "estimate_delta_random_forest_minus_xgboost": float(full_rf[name] - full_xgb[name]),
                    "ci95_low": float(np.nanpercentile(finite, 2.5)) if len(finite) else np.nan,
                    "ci95_high": float(np.nanpercentile(finite, 97.5)) if len(finite) else np.nan,
                    "two_sided_bootstrap_p": float(min(1.0, 2 * min(np.mean(finite <= 0), np.mean(finite >= 0)))) if len(finite) else np.nan,
                    "bootstrap_replicates": n_boot,
                    "resampling_unit": "patient_group",
                    "matched_images": int(len(paired)),
                    "threshold_source": "validation_locked_per_classifier_head",
                    "shared_resnet_features": True,
                })
    if not metric_frames or not prediction_frames:
        raise RuntimeError("No completed RF-vs-XGBoost classifier-head ablations were found")
    pd.concat(metric_frames, ignore_index=True).to_csv(output / "classifier_head_ablation_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output / "classifier_head_ablation_predictions.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(output / "classifier_head_ablation_paired_bootstrap.csv", index=False)


def completed_run_directories(output: Path, seed: int) -> list[Path]:
    seed_root = output / "runs" / f"seed_{seed}"
    completed = []
    if not seed_root.exists():
        return completed
    for marker in sorted(seed_root.rglob("run_complete.json")):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            manifest_path = marker.parent / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        required = [
            marker.parent / "predictions.csv",
            marker.parent / "metrics_operating_points.csv",
            marker.parent / "training_manifest.csv",
            marker.parent / "thresholds.json",
            marker.parent / "history.csv",
            marker.parent / "training_diagnostics.json",
            marker.parent / "backbone_best_state.pt",
            marker.parent / "xgboost_model.ubj",
        ]
        parsed_route = route_metadata(str(manifest.get("route", "")))
        if (
            parsed_route is not None
            and parsed_route["mode"] == "primary_gan_resnet50_xgb"
            and parsed_route["strategy"] == "original_ratio"
            and parsed_route["ratio"] in {0.0, 4.0}
            and parsed_route["noise"] == 0.0
        ):
            required.extend([
                marker.parent / "classifier_head_ablation" / "metrics_operating_points.csv",
                marker.parent / "classifier_head_ablation" / "predictions.csv",
                marker.parent / "classifier_head_ablation" / "thresholds.json",
                marker.parent / "classifier_head_ablation" / "random_forest_model.joblib",
                marker.parent / "classifier_head_ablation" / "ablation_complete.json",
            ])
        if (
            payload.get("status") == "complete"
            and payload.get("configuration_fingerprint") == manifest.get("configuration_fingerprint")
            and manifest.get("code_sha256") == sha256_file(Path(__file__).resolve())
            and int(manifest.get("seed", -1)) == int(seed)
            and all(path.is_file() and path.stat().st_size > 0 for path in required)
        ):
            completed.append(marker.parent)
    return completed


def build_clean_workbook(output: Path):
    import openpyxl

    workbook_path = output / "sycnn_lidsentry_final_results.xlsx"
    run_dirs = completed_run_directories(output, int(CFG.get("seed", SEEDS[0])))
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        metric_files = [run_dir / "metrics_operating_points.csv" for run_dir in run_dirs]
        pred_files = [run_dir / "predictions.csv" for run_dir in run_dirs]
        frames = []
        for path in metric_files:
            frames.append(pd.read_csv(path))
        if frames:
            all_metrics = pd.concat(frames, ignore_index=True)
            all_metrics.to_excel(writer, sheet_name="Operating_points", index=False)
            all_metrics[all_metrics["operating_point"] == "youden"].to_excel(writer, sheet_name="Main_results", index=False)
            all_metrics[all_metrics["route"].str.contains("_lidsentry_", regex=False)].to_excel(writer, sheet_name="LidSentry_ablation", index=False)
        if pred_files:
            pd.concat([pd.read_csv(path) for path in pred_files], ignore_index=True).to_excel(writer, sheet_name="Raw_predictions", index=False)
        ci = output / "final_95ci_patient_group_bootstrap.csv"
        if ci.exists():
            pd.read_csv(ci).to_excel(writer, sheet_name="CI_95_patient_group", index=False)
        paired = output / "paired_patient_group_bootstrap_deltas.csv"
        if paired.exists():
            try:
                pd.read_csv(paired).to_excel(writer, sheet_name="Paired_deltas", index=False)
            except pd.errors.EmptyDataError:
                pass
        interaction = output / "factorial_2x2_interaction_bootstrap.csv"
        if interaction.exists():
            pd.read_csv(interaction).to_excel(writer, sheet_name="Factorial_interaction", index=False)
        head_metrics = output / "classifier_head_ablation_metrics.csv"
        if head_metrics.exists():
            pd.read_csv(head_metrics).to_excel(writer, sheet_name="Head_ablation", index=False)
        head_ci = output / "classifier_head_ablation_paired_bootstrap.csv"
        if head_ci.exists():
            pd.read_csv(head_ci).to_excel(writer, sheet_name="Head_ablation_CI", index=False)
        head_predictions = output / "classifier_head_ablation_predictions.csv"
        if head_predictions.exists():
            pd.read_csv(head_predictions).to_excel(writer, sheet_name="Head_ablation_predictions", index=False)
        split = output / "split_manifest.csv"
        if split.exists():
            pd.read_csv(split).drop(
                columns=["absolute_path", "relative_path", "filename"], errors="ignore"
            ).to_excel(writer, sheet_name="Split_audit", index=False)
        duplicate = output / "near_duplicate_audit.csv"
        if duplicate.exists():
            pd.read_csv(duplicate).to_excel(writer, sheet_name="Near_duplicates", index=False)
        sim_files = [run_dir / "similarity" / "synthetic_real_nearest_neighbors.csv" for run_dir in run_dirs if (run_dir / "similarity" / "synthetic_real_nearest_neighbors.csv").is_file()]
        if sim_files:
            pd.concat([pd.read_csv(path) for path in sim_files], ignore_index=True).drop(
                columns=["synthetic_path", "nearest_real_path"], errors="ignore"
            ).to_excel(writer, sheet_name="Synthetic_similarity", index=False)
        duplicate_flag_files = [run_dir / "similarity" / "synthetic_real_duplicate_flags.csv" for run_dir in run_dirs if (run_dir / "similarity" / "synthetic_real_duplicate_flags.csv").is_file()]
        if duplicate_flag_files:
            pd.concat([pd.read_csv(path) for path in duplicate_flag_files], ignore_index=True).drop(
                columns=["synthetic_path"], errors="ignore"
            ).to_excel(writer, sheet_name="Synthetic_duplicates", index=False)
        self_duplicate_files = [run_dir / "similarity" / "synthetic_synthetic_duplicate_flags.csv" for run_dir in run_dirs if (run_dir / "similarity" / "synthetic_synthetic_duplicate_flags.csv").is_file()]
        if self_duplicate_files:
            pd.concat([pd.read_csv(path) for path in self_duplicate_files], ignore_index=True).drop(
                columns=["synthetic_path"], errors="ignore"
            ).to_excel(writer, sheet_name="Synthetic_self_duplicates", index=False)
        calibration = output / "calibration_bins.csv"
        if calibration.exists():
            pd.read_csv(calibration).to_excel(writer, sheet_name="Calibration_bins", index=False)
        decision = output / "decision_curve_net_benefit.csv"
        if decision.exists():
            pd.read_csv(decision).to_excel(writer, sheet_name="Decision_curve", index=False)
        traceability_rows = []
        for run_dir in run_dirs:
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            traceability_rows.append({
                "route": manifest.get("route"),
                "seed": manifest.get("seed"),
                "ratio": manifest.get("ratio"),
                "input_size": manifest.get("input_size"),
                "backbone": manifest.get("backbone"),
                "lidsentry": manifest.get("lidsentry"),
                "class_strategy": manifest.get("class_strategy"),
                "label_noise_rate": manifest.get("label_noise_rate"),
                "strong_baseline": manifest.get("strong_baseline", False),
                "pretrained_weights_used": manifest.get("pretrained_weights_used"),
                "class_weighted_loss": manifest.get("class_weighted_loss"),
                "training_augmentation": ";".join(manifest.get("training_augmentation", [])),
                "quality_aware_components": ";".join(manifest.get("quality_aware_components", [])),
                "synthetic_sampling_with_replacement": manifest.get("synthetic_sampling_with_replacement"),
                "configuration_fingerprint": manifest.get("configuration_fingerprint"),
            })
        if traceability_rows:
            pd.DataFrame(traceability_rows).to_excel(writer, sheet_name="Traceability", index=False)
        diagnostic_rows = []
        for run_dir in run_dirs:
            diagnostics_path = run_dir / "training_diagnostics.json"
            if diagnostics_path.is_file():
                diagnostic_rows.append(json.loads(diagnostics_path.read_text(encoding="utf-8")))
        if diagnostic_rows:
            pd.DataFrame(diagnostic_rows).to_excel(writer, sheet_name="Training_diagnostics", index=False)
        gan_history_path = output / "synthetic" / f"seed_{int(CFG.get('seed', SEEDS[0]))}" / "gan_history.csv"
        if gan_history_path.is_file():
            pd.read_csv(gan_history_path).to_excel(writer, sheet_name="GAN_training", index=False)
    workbook = openpyxl.load_workbook(workbook_path)
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
    workbook.save(workbook_path)
    print(f"[INFO] wrote clean workbook {workbook_path}")


def main():
    seed = int(CFG.get("seed", SEEDS[0]))
    seed_everything(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    resolved_runtime = {
        "stage": str(CFG.get("stage", "audit")).lower(),
        "seed": seed,
        "ratios": [float(value) for value in CFG.get("ratios", RATIOS)],
        "image_size": IMAGE_SIZE,
        "gan_size": GAN_SIZE,
        "gan_batch_size": GAN_BATCH_SIZE,
        "backbone_initialization": BACKBONE_INIT,
        "epochs": EPOCHS,
        "gan_epochs": GAN_EPOCHS,
        "gan_patience": GAN_PATIENCE,
        "gan_min_epochs": GAN_MIN_EPOCHS,
        "gan_min_delta": GAN_MIN_DELTA,
        "patience": PATIENCE,
        "cnn_min_epochs": CNN_MIN_EPOCHS,
        "cnn_min_delta": CNN_MIN_DELTA,
        "batch_size": BATCH_SIZE,
        "bootstrap_replicates": N_BOOT,
        "num_workers": NUM_WORKERS,
        "require_cuda": REQUIRE_CUDA,
        "device": device_name(),
        "data_root": str(DATA_ROOT),
        "output_root": str(OUT),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "package_versions": package_versions(),
    }
    (OUT / "resolved_runtime.json").write_text(json.dumps(resolved_runtime, indent=2), encoding="utf-8")
    print(json.dumps(resolved_runtime, indent=2))
    (OUT / "phase.txt").write_text("starting archive discovery\n", encoding="utf-8")
    canonical_root = os.environ.get("CANONICAL_IMAGE_ROOT")
    if canonical_root:
        data_dir = Path(canonical_root)
    else:
        archive = find_zip()
        data_dir = safe_extract(archive, DATA_ROOT)
    (OUT / "phase.txt").write_text("archive extracted; loading locked manifest\n", encoding="utf-8")
    ensure_locked_assets()
    manifest = load_precomputed_manifest(data_dir)
    if manifest is None:
        manifest = build_manifest(data_dir)
    (OUT / "phase.txt").write_text(f"manifest ready: {len(manifest)} rows; loading locked split\n", encoding="utf-8")
    frame, split_audit = load_locked_split(manifest)
    (OUT / "phase.txt").write_text("locked split ready; audit outputs written\n", encoding="utf-8")
    frame.to_csv(OUT / "split_manifest.csv", index=False)
    (OUT / "split_integrity_summary.json").write_text(json.dumps(split_audit, indent=2), encoding="utf-8")
    save_near_duplicate_audit(frame, OUT)
    print(json.dumps(split_audit, indent=2))
    stage = str(CFG.get("stage", "audit")).lower()
    active_ratios = [float(value) for value in CFG.get("ratios", RATIOS)]
    if stage == "audit":
        return
    train = frame[frame["split"] == "train"].copy()
    needs_synthetic = (
        stage in {"primary_ablation", "balanced_ablation"}
        or any(float(value) > 0 for value in active_ratios)
    )
    pool_dir = train_gan(train, seed) if stage in {"legacy", "primary", "primary_sweep", "primary_ablation", "balanced_ablation", "full", "generate"} and needs_synthetic else None
    if stage == "generate":
        return
    if stage in {"legacy", "full"}:
        active_strategies = [str(value) for value in CFG.get("strategies", ["original_ratio", "balanced"])]
        for strategy in active_strategies:
            for ratio in active_ratios:
                run_route(frame, split_audit, seed, ratio, 224, False, strategy, "legacy_gan_resnet50_xgb", pool_dir)
    if stage in {"primary", "primary_sweep", "full"}:
        strategies = ["original_ratio", "balanced"] if stage in {"primary", "full"} else ["original_ratio"]
        qualities = [False, True]
        for active_strategy in strategies:
            for ratio in active_ratios:
                for quality in qualities:
                    run_route(frame, split_audit, seed, ratio, 224, quality, active_strategy, "primary_gan_resnet50_xgb", pool_dir)
        if stage in {"primary", "primary_sweep", "full"} and bool(CFG.get("include_strong_baseline", True)):
            run_route(frame, split_audit, seed, 0, 224, False, "original_ratio", "strong_baseline_resnet50", None, backbone="resnet50")
        if bool(CFG.get("include_source_holdout", False)):
            run_route(frame, split_audit, seed, 0, 224, False, "original_ratio", "source_holdout", None, source_holdout=True)
    if stage in {"primary_ablation", "full"}:
        for strategy in ["original_ratio", "balanced"]:
            for quality in [False, True]:
                run_route(frame, split_audit, seed, 4, 224, quality, strategy, "primary_gan_resnet50_xgb", pool_dir)
    if stage == "balanced_ablation":
        for quality in [False, True]:
            run_route(frame, split_audit, seed, 4, 224, quality, "balanced",
                      "primary_gan_resnet50_xgb", pool_dir)
    if stage == "strong_baseline":
        run_route(frame, split_audit, seed, 0, 224, False, "original_ratio",
                  "strong_baseline_resnet50", None, backbone="resnet50")
    if stage == "robustness":
        noise_rates = [float(value.strip()) for value in os.environ.get("RUN_NOISE_RATES", "0.05,0.10").split(",") if value.strip()]
        for label_noise in noise_rates:
            for quality in [False, True]:
                run_route(frame, split_audit, seed, 0, 224, quality, "original_ratio", "label_noise_robustness", None, label_noise=label_noise)
    if not SKIP_POSTPROCESS and stage in {"legacy", "primary", "primary_sweep", "primary_ablation", "balanced_ablation", "robustness", "analyze", "strong_baseline", "full"}:
        completed_runs = completed_run_directories(OUT, seed)
        predictions = [run_dir / "predictions.csv" for run_dir in completed_runs]
        if not predictions:
            raise RuntimeError("No completed prediction routes are available for post-processing")
        bootstrap_ci(predictions, OUT, N_BOOT)
        paired_bootstrap_ci(predictions, OUT, N_BOOT)
        factorial_interaction_bootstrap(predictions, OUT, N_BOOT)
        classifier_head_ablation_summary(completed_runs, OUT, N_BOOT)
        save_calibration_and_decision_curve(predictions, OUT)
        build_clean_workbook(OUT)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
