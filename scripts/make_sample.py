"""
Generate a synthetic nadir scene with known building heights and physically
correct cast shadows, written as a proper GeoTIFF with sun-geometry tags.

Why this exists: real imagery with usable sun metadata takes time to source,
and you have two days. This gives you a scene where you know the true height
of every building, so you can unit-test the shadow formula

    h = L_pixels * GSD * tan(sun_elevation)

and prove your calibration is correct before you ever touch real data. Keep
it in the repo permanently as a regression test -- it is exactly the datum /
convention unit test the brief calls for.

    python scripts/make_sample.py --out data/raw/synthetic.tif
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


# (row, col, width_px, depth_px, height_m)
BUILDINGS = [
    (120, 150, 60, 60, 12.0),
    (140, 420, 80, 55, 28.0),
    (330, 200, 50, 50, 45.0),
    (300, 620, 70, 70, 18.0),
    (560, 350, 90, 60, 34.0),
    (600, 700, 45, 45, 55.0),
    (700, 160, 65, 80, 22.0),
    (450, 830, 55, 55, 40.0),
]

GROUND_RGB = (118, 112, 104)
SHADOW_RGB = (34, 33, 38)
ROOF_PALETTE = [(196, 188, 176), (150, 120, 105), (170, 172, 178),
                (132, 140, 130), (205, 200, 190)]


def build_scene(size: int, gsd: float, sun_elev: float, sun_azim: float):
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:, :] = GROUND_RGB

    # Faint texture so the depth model has something to grip and so shadow
    # thresholding is not trivially easy.
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 7, (size, size, 1))
    rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    truth_grid = np.zeros((size, size), dtype=np.float32)

    # Shadows fall in the direction opposite the sun.
    shadow_azimuth = math.radians((sun_azim + 180.0) % 360.0)
    east = math.sin(shadow_azimuth)
    north = math.cos(shadow_azimuth)
    tan_elev = math.tan(math.radians(sun_elev))

    # Pass 1: shadows (drawn first so roofs paint over them)
    for row, col, w, d, height in BUILDINGS:
        length_px = (height / tan_elev) / gsd
        steps = max(2, int(length_px * 2))
        for i in range(steps + 1):
            travel = length_px * (i / steps)
            dr = int(round(-north * travel))   # row grows southward
            dc = int(round(east * travel))
            r0, r1 = max(0, row + dr), min(size, row + dr + d)
            c0, c1 = max(0, col + dc), min(size, col + dc + w)
            if r0 < r1 and c0 < c1:
                rgb[r0:r1, c0:c1] = SHADOW_RGB

    # Pass 2: roofs + ground-truth height grid
    for idx, (row, col, w, d, height) in enumerate(BUILDINGS):
        r1, c1 = min(size, row + d), min(size, col + w)
        roof = np.array(ROOF_PALETTE[idx % len(ROOF_PALETTE)], dtype=np.float32)
        shade = rng.normal(0, 5, (r1 - row, c1 - col, 1))
        rgb[row:r1, col:c1] = np.clip(roof + shade, 0, 255).astype(np.uint8)
        truth_grid[row:r1, col:c1] = height

    return rgb, truth_grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/synthetic.tif"))
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--gsd", type=float, default=0.5, help="metres per pixel")
    ap.add_argument("--sun-elev", type=float, default=42.0)
    ap.add_argument("--sun-azim", type=float, default=135.0)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rgb, truth = build_scene(args.size, args.gsd, args.sun_elev, args.sun_azim)

    # UTM 44N, a plausible Indian AOI. Projected CRS -> GSD is already metres.
    transform = from_origin(500_000.0, 1_800_000.0, args.gsd, args.gsd)

    with rasterio.open(
        args.out, "w", driver="GTiff",
        height=args.size, width=args.size, count=3,
        dtype="uint8", crs="EPSG:32644", transform=transform,
    ) as dst:
        for band in range(3):
            dst.write(rgb[:, :, band], band + 1)
        dst.update_tags(
            SUN_ELEVATION=f"{args.sun_elev}",
            SUN_AZIMUTH=f"{args.sun_azim}",
            ACQUISITION_DATE="2026-08-26",
        )

    truth_path = args.out.with_suffix(".truth.npy")
    np.save(truth_path, truth)

    meta_path = args.out.with_suffix(".truth.json")
    with open(meta_path, "w") as fh:
        json.dump({
            "gsd_m": args.gsd,
            "sun_elevation_deg": args.sun_elev,
            "sun_azimuth_deg": args.sun_azim,
            "buildings": [
                {"id": i, "row": r, "col": c, "w": w, "d": d,
                 "height_m": h,
                 "expected_shadow_px": (h / math.tan(math.radians(args.sun_elev))) / args.gsd}
                for i, (r, c, w, d, h) in enumerate(BUILDINGS)
            ],
        }, fh, indent=2)

    print(f"wrote {args.out}")
    print(f"      {truth_path}   (per-pixel true height, your nDSM ground truth)")
    print(f"      {meta_path}    (per-building truth + expected shadow lengths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
