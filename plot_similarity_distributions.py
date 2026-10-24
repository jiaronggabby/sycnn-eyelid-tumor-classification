#!/usr/bin/env python3
"""Export identifier-free similarity metrics and plot their distributions.

The input is the per-sample output of ``similarity_analysis.py``. The exported
CSV deliberately contains only method, class, and numeric similarity/duplicate
fields; patient identifiers, image identifiers, and file paths are not copied.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from statistics import median
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


METHOD_ORDER = ("gan", "diffusion")
COLORS = {
    "gan": {"point": (226, 153, 127, 160), "box": (237, 194, 178, 210)},
    "diffusion": {"point": (105, 157, 187, 160), "box": (165, 196, 214, 210)},
}
INK = (36, 43, 51, 255)
GRID = (221, 227, 233, 255)
RESAMPLING = getattr(Image, "Resampling", Image)

NUMERIC_FIELDS = (
    "generation_method",
    "synthetic_label",
    "feature_cosine_similarity",
    "ssim_to_nearest_feature_neighbor",
    "minimum_fit_dhash_hamming",
    "dhash_near_duplicate_le_2",
    "exact_canonical_pixel_match",
)


def load_rows(path: Path, expected_per_method: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(NUMERIC_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input is missing columns: {', '.join(sorted(missing))}")
        rows = []
        for row in reader:
            method = row["generation_method"].strip().lower()
            if method not in METHOD_ORDER:
                raise ValueError(f"Unexpected generation method: {method!r}")
            rows.append({field: row[field] for field in NUMERIC_FIELDS})
    if not rows:
        raise ValueError("Input contains no sample rows")
    counts = {method: sum(r["generation_method"] == method for r in rows) for method in METHOD_ORDER}
    if any(counts[method] != expected_per_method for method in METHOD_ORDER):
        raise ValueError(f"Expected {expected_per_method} samples per method, found {counts}")
    return rows


def export_numeric_csv(rows: Iterable[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=NUMERIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("No supported font file was found; pass --font explicitly")


def fonts(font_path: Path | None) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    regular = font_path or first_existing(
        [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    bold = first_existing(
        [
            regular.with_name("arialbd.ttf"),
            regular.with_name("LiberationSans-Bold.ttf"),
            regular.with_name("DejaVuSans-Bold.ttf"),
            regular,
        ]
    )
    return ImageFont.truetype(str(regular), 58), ImageFont.truetype(str(bold), 72)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def draw_panel(
    image: Image.Image,
    rows: list[dict[str, str]],
    metric_column: str,
    axis_label: str,
    label: str,
    x0: int,
    fonts_pair: tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont],
    seed: int,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    regular, bold = fonts_pair
    panel_width = 1940
    axis_left = x0 + 220
    axis_right = x0 + panel_width - 65
    axis_top = 205
    axis_bottom = 1460
    axis_height = axis_bottom - axis_top

    def y(value: float) -> int:
        return round(axis_bottom - value * axis_height)

    # Panel letter sits in the reserved upper-left margin, outside the axes.
    draw.text((x0 + 105, 76), label, font=bold, fill=INK)

    for tick in range(6):
        value = tick / 5
        yy = y(value)
        draw.line((axis_left, yy, axis_right, yy), fill=GRID, width=3)
        tick_box = draw.textbbox((0, 0), f"{value:.1f}", font=regular)
        draw.text((axis_left - 28 - (tick_box[2] - tick_box[0]), yy - 34), f"{value:.1f}", font=regular, fill=INK)

    draw.line((axis_left, axis_top, axis_left, axis_bottom), fill=INK, width=5)
    draw.line((axis_left, axis_bottom, axis_right, axis_bottom), fill=INK, width=5)

    rotated = Image.new("RGBA", (1100, 100), (255, 255, 255, 0))
    ImageDraw.Draw(rotated).text((0, 0), axis_label, font=regular, fill=INK)
    rotated = rotated.rotate(90, expand=True, resample=RESAMPLING.BICUBIC)
    image.alpha_composite(rotated, (x0 + 20, axis_top + (axis_height - rotated.height) // 2))
    draw = ImageDraw.Draw(image, "RGBA")

    centers = (axis_left + round((axis_right - axis_left) * 0.31), axis_left + round((axis_right - axis_left) * 0.71))
    jitter = random.Random(seed)
    for method, center in zip(METHOD_ORDER, centers):
        group = [float(row[metric_column]) for row in rows if row["generation_method"].lower() == method]
        expected_count = sum(r["generation_method"].lower() == method for r in rows)
        if len(group) != expected_count:
            raise ValueError(f"Inconsistent values for {method} in {metric_column}")

        q1 = quantile(group, 0.25)
        med = median(group)
        q3 = quantile(group, 0.75)
        spread = q3 - q1
        lower_fence = q1 - 1.5 * spread
        upper_fence = q3 + 1.5 * spread
        inliers = [v for v in group if lower_fence <= v <= upper_fence]
        whisker_low = min(inliers)
        whisker_high = max(inliers)
        box_width = 300
        color = COLORS[method]

        draw.line((center, y(whisker_low), center, y(whisker_high)), fill=INK, width=5)
        draw.line((center - 90, y(whisker_low), center + 90, y(whisker_low)), fill=INK, width=5)
        draw.line((center - 90, y(whisker_high), center + 90, y(whisker_high)), fill=INK, width=5)
        draw.rectangle((center - box_width // 2, y(q3), center + box_width // 2, y(q1)), fill=color["box"], outline=INK, width=4)
        draw.line((center - box_width // 2, y(med), center + box_width // 2, y(med)), fill=INK, width=6)

        for value in group:
            xx = center + jitter.randint(-145, 145)
            yy = y(value)
            draw.ellipse((xx - 12, yy - 12, xx + 12, yy + 12), fill=color["point"])

        display_name = "GAN" if method == "gan" else "Diffusion"
        label_box = draw.textbbox((0, 0), display_name, font=regular)
        draw.text((center - (label_box[2] - label_box[0]) // 2, axis_bottom + 48), display_name, font=regular, fill=INK)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="per_sample_neighbors.csv from the analysis")
    parser.add_argument("--numeric-csv", type=Path, required=True, help="identifier-free metric table to export")
    parser.add_argument("--figure", type=Path, required=True, help="output PNG for Supplementary Figure S4")
    parser.add_argument("--expected-per-method", type=int, default=200)
    parser.add_argument("--font", type=Path, help="optional path to a TrueType sans-serif font")
    args = parser.parse_args()

    if args.expected_per_method < 1:
        parser.error("--expected-per-method must be positive")
    rows = load_rows(args.input_csv, args.expected_per_method)
    export_numeric_csv(rows, args.numeric_csv)

    width, height = 4200, 1800
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    font_pair = fonts(args.font)
    draw_panel(image, rows, "feature_cosine_similarity", "ResNet18 cosine similarity", "A", 0, font_pair, 2401)
    draw_panel(image, rows, "ssim_to_nearest_feature_neighbor", "SSIM", "B", 2070, font_pair, 2402)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(args.figure, dpi=(600, 600), optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
