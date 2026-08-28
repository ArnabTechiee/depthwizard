"""
DepthWizard - Stage 4: bake viewer assets.

The demo must never run inference. Everything the browser needs is written
here as static files, so the viewer opens instantly, works offline, and
cannot crash in front of a panel because a model failed to load.

    python -m pipeline.bake antakya
    python -m pipeline.bake antakya --grid 384
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def resample(arr: np.ndarray, size: int) -> np.ndarray:
    rows = np.linspace(0, arr.shape[0] - 1, size).round().astype(int)
    cols = np.linspace(0, arr.shape[1] - 1, size).round().astype(int)
    return arr[np.ix_(rows, cols)]


def run(scene: str, work_root: Path, out_root: Path, grid: int) -> dict:
    work_dir = work_root / scene
    out_dir = out_root / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((work_dir / "ingest.json").read_text())
    heights = np.load(work_dir / "ndsm_m.npy").astype(np.float32)
    rgb = np.load(work_dir / "rgb.npy")

    # A 512x512 grid is ~262k vertices: plenty of detail, and it renders at
    # 60fps in a browser without any LOD machinery. Full resolution would be
    # millions of vertices and would stall the tab.
    grid = min(grid, min(heights.shape))
    h_small = resample(heights, grid)
    h_small = np.nan_to_num(h_small, nan=0.0, posinf=0.0, neginf=0.0)
    h_small = np.maximum(h_small, 0.0)

    (out_dir / "heights.bin").write_bytes(h_small.astype("<f4").tobytes())

    tex_size = min(1024, max(rgb.shape[0], 512))
    Image.fromarray(rgb).resize((tex_size, tex_size), Image.LANCZOS) \
        .save(out_dir / "texture.jpg", quality=88)

    buildings = []
    stats = {}
    bpath = work_dir / "buildings.json"
    if bpath.exists():
        data = json.loads(bpath.read_text())
        stats = {k: v for k, v in data.items() if k != "buildings"}
        for b in data.get("buildings", []):
            if b.get("height_m") is None:
                continue
            buildings.append({
                "id": b["id"],
                "x": round(b["col"] / heights.shape[1], 5),
                "y": round(b["row"] / heights.shape[0], 5),
                "h": b["height_m"],
                "area": b.get("area_m2"),
                "m": b.get("method", "shadow"),
                "r": b.get("repeatability_m"),
            })

    # Fold in LiDAR validation if pipeline.validate has been run. Repeatability
    # is a precision measure -- it says the method is self-consistent, not that
    # it is right. These are the accuracy numbers, against independent ground
    # truth, and they belong on screen rather than buried in a JSON file.
    vpath = work_dir / "validation.json"
    validation_line = None
    if vpath.exists():
        v = json.loads(vpath.read_text())
        pb = v.get("per_building") or {}
        if pb:
            stats["lidar_rmse_m"] = pb.get("rmse_m")
            stats["lidar_mae_m"] = pb.get("mae_m")
            stats["lidar_bias_m"] = pb.get("bias_m")
            stats["lidar_corr"] = pb.get("correlation")
            stats["lidar_n"] = pb.get("n_buildings")
            validation_line = (f"RMSE {pb.get('rmse_m')} m vs LiDAR "
                               f"(n={pb.get('n_buildings')})")
        stats["lidar_source"] = v.get("lidar_source")
        stats["lidar_coverage"] = v.get("dsm_coverage")

    payload = {
        "scene": scene,
        "grid": grid,
        "gsd_m": meta["gsd_m"],
        "span_m": round(meta["gsd_m"] * heights.shape[1], 1),
        "crs": meta["crs"],
        "sun_elevation_deg": meta["sun_elevation_deg"],
        "sun_azimuth_deg": meta["sun_azimuth_deg"],
        "acquisition_date": meta.get("acquisition_date"),
        "max_height_m": round(float(h_small.max()), 2),
        "stats": stats,
        "buildings": buildings,
    }
    (out_dir / "meta.json").write_text(json.dumps(payload, indent=1))

    print(f"  {scene}")
    print(f"    grid       : {grid} x {grid}  ({grid * grid} vertices)")
    print(f"    span       : {payload['span_m']} m across")
    print(f"    max height : {payload['max_height_m']} m")
    print(f"    buildings  : {len(buildings)}")
    if validation_line:
        print(f"    validation : {validation_line}")
    print(f"    -> {out_dir}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--work", type=Path, default=Path("data/work"))
    ap.add_argument("--out", type=Path, default=Path("viewer/data"))
    ap.add_argument("--grid", type=int, default=512)
    args = ap.parse_args()

    index = []
    for scene in args.scenes:
        index.append(run(scene, args.work, args.out, args.grid))

    (args.out / "index.json").write_text(json.dumps(
        [{"scene": p["scene"], "max_height_m": p["max_height_m"],
          "buildings": len(p["buildings"])} for p in index], indent=1))
    print(f"\n  wrote {args.out / 'index.json'} ({len(index)} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
