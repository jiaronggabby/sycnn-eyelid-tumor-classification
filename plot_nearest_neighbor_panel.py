#!/usr/bin/env python3
"""Create an ID-free panel of the most similar synthetic-to-training pairs.

Private manifests and image files are read locally on the approved research
server. The exported figure contains pixels only, without IDs, paths or metadata.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


INK = (35, 43, 52)
MUTED = (92, 103, 114)
LINE = (190, 198, 205)
METHODS = (("gan", "GAN"), ("diffusion", "Diffusion"))
CLASSES = ((0, "Benign"), (1, "Malignant"))
RESAMPLING = getattr(Image, "Resampling", Image)


def font(size: int, bold: bool = False, requested: Path | None = None) -> ImageFont.FreeTypeFont:
    candidates = [requested] if requested else []
    candidates.extend([
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ])
    for candidate in candidates:
        if candidate and candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def load_rgb(path: str, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    image.thumbnail(size, RESAMPLING.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("per_sample_csv", type=Path)
    parser.add_argument("samples_csv", type=Path, help="private selected synthetic manifest")
    parser.add_argument("training_reference_csv", type=Path, help="private real training reference manifest")
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()

    with args.per_sample_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        neighbor_rows = list(csv.DictReader(stream))
    with args.samples_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        samples = list(csv.DictReader(stream))
    with args.training_reference_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        references = list(csv.DictReader(stream))

    sample_lookup = {
        (row["generation_method"].lower(), str(row["seed"]), row["synthetic_id"]): row
        for row in samples
    }
    reference_lookup = {row["image_id"]: row for row in references}
    selected = []
    for method, method_name in METHODS:
        for label, label_name in CLASSES:
            candidates = [
                row for row in neighbor_rows
                if row["generation_method"].lower() == method and int(row["synthetic_label"]) == label
            ]
            if not candidates:
                raise ValueError(f"No rows for {method_name}, {label_name}")
            best = max(candidates, key=lambda row: float(row["feature_cosine_similarity"]))
            sample_key = (method, str(best["seed"]), best["synthetic_id"])
            if sample_key not in sample_lookup:
                raise ValueError(f"Synthetic image is missing from the private manifest: {sample_key}")
            reference_id = best["nearest_training_image_id"]
            if reference_id not in reference_lookup:
                raise ValueError(f"Nearest training image is missing from the private manifest: {reference_id}")
            selected.append((method_name, label_name, best, sample_lookup[sample_key], reference_lookup[reference_id]))

    width, height = 3000, 3400
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    heading = font(62, True, args.font)
    subheading = font(56, True, args.font)
    body = font(54, False, args.font)
    margin_x, label_w, gap, img_w, img_h = 80, 310, 45, 1120, 680
    col1 = margin_x + label_w + gap
    col2 = col1 + img_w + 150
    draw.text((col1 + 280, 45), "Synthetic image", font=heading, fill=INK)
    draw.text((col2 + 190, 45), "Nearest real training image", font=heading, fill=INK)

    row_top = 145
    row_gap = 780
    for index, (method_name, label_name, match, sample, reference) in enumerate(selected):
        top = row_top + index * row_gap
        draw.text((margin_x, top + 160), f"{method_name}\n{label_name}", font=subheading, fill=INK, spacing=14)
        synth = load_rgb(sample["synthetic_path"], (img_w, img_h))
        real = load_rgb(reference["absolute_path"], (img_w, img_h))
        image.paste(synth, (col1, top))
        image.paste(real, (col2, top))
        draw = ImageDraw.Draw(image)
        draw.rectangle((col1, top, col1 + img_w, top + img_h), outline=LINE, width=4)
        draw.rectangle((col2, top, col2 + img_w, top + img_h), outline=LINE, width=4)
        nearest_class = "benign" if int(reference["label"]) == 0 else "malignant"
        details = (
            f"Cosine: {float(match['feature_cosine_similarity']):.3f}    "
            f"SSIM: {float(match['ssim_to_nearest_feature_neighbor']):.3f}"
        )
        draw.text((col1, top + img_h + 14), details, font=body, fill=MUTED)
        draw.text((col2, top + img_h + 14), f"Reference class: {nearest_class}", font=body, fill=MUTED)
        if index < len(selected) - 1:
            draw.line((margin_x, top + row_gap - 30, width - margin_x, top + row_gap - 30), fill=(232, 235, 238), width=2)

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.figure, format="PNG", dpi=(600, 600), optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
