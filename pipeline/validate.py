"""
DepthWizard - Phase 1: validate against LiDAR ground truth.

Turns "repeatability 3.1 m" (a precision measure) into "RMSE X m against
LiDAR" (an accuracy measure) -- which is what 50% of the PS rubric asks for.

Reference construction from a raw point cloud:

    DSM  = highest return per cell          (roofs, canopy)
    DTM  = lowest GROUND-classified return  (bare earth, ASPRS class 2)
    nDSM = DSM - DTM                        (height above ground)

Two things worth knowing:

1. We compare nDSM to nDSM, never absolute elevation. Height-above-ground is
   datum-independent, so the EGM96-vs-WGS84 offset that would otherwise inject
   tens of metres cancels out entirely. Do not "fix" this by comparing DSMs.

2. Downloadable 3DEP *DEM* products are bare earth -- buildings already
   removed. They cannot validate building heights. You need the point cloud.

    python -m pipeline.validate antakya --lidar data/lidar/points.laz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

GROUND_CLASS = 2          # ASPRS standard classification for bare earth


def load_points(laz_paths, target_crs: str):
    """Read one or more LAS/LAZ files and reproject XY into the scene's CRS."""
    try:
        import laspy
    except ImportError:
        raise SystemExit(
            "error: laspy is required.\n"
            "  pip install \"laspy[lazrs]\"\n"
            "  (the [lazrs] extra is what reads compressed .laz)"
        )

    xs, ys, zs, cs = [], [], [], []
    for laz_path in laz_paths:
        print(f"  reading {laz_path.name} ...")
        with laspy.open(str(laz_path)) as fh:
            las = fh.read()

        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        z = np.asarray(las.z, dtype=np.float64)
        cls = np.asarray(las.classification, dtype=np.uint8)

        src_crs = None
        try:
            src_crs = las.header.parse_crs()
        except Exception:
            pass

        if src_crs is not None and str(src_crs) != str(target_crs):
            from pyproj import Transformer
            tr = Transformer.from_crs(src_crs, target_crs, always_xy=True)
            x, y = tr.transform(x, y)
        elif src_crs is None:
            print("  ! no CRS in file; assuming it already matches the scene")

        print(f"    {x.size:,} points")
        xs.append(x); ys.append(y); zs.append(z); cs.append(cls)

    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(zs), np.concatenate(cs))


def rasterise(x, y, z, transform, shape, reducer="max"):
    """Bin points onto the scene grid, keeping max or min z per cell."""
    inv = ~transform
    cols, rows = inv * (x, y)
    cols = np.floor(cols).astype(np.int64)
    rows = np.floor(rows).astype(np.int64)

    inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    if not inside.any():
        return None, 0
    rows, cols, z = rows[inside], cols[inside], z[inside]

    flat = rows * shape[1] + cols
    grid = np.full(shape[0] * shape[1], np.nan, dtype=np.float32)

    # np.maximum.at / minimum.at handle duplicate indices correctly
    seed = -np.inf if reducer == "max" else np.inf
    acc = np.full(grid.size, seed, dtype=np.float64)
    (np.maximum.at if reducer == "max" else np.minimum.at)(acc, flat, z)
    filled = np.isfinite(acc)
    grid[filled] = acc[filled].astype(np.float32)

    return grid.reshape(shape), int(inside.sum())


def fill_gaps(grid: np.ndarray) -> np.ndarray:
    """Nearest-neighbour fill for cells with no returns."""
    holes = ~np.isfinite(grid)
    if not holes.any():
        return grid
    idx = ndimage.distance_transform_edt(holes, return_distances=False,
                                         return_indices=True)
    return grid[tuple(idx)]


def metrics(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> dict:
    p, r = pred[mask], ref[mask]
    if p.size < 10:
        return {"n": int(p.size)}
    err = p - r
    out = {
        "n": int(p.size),
        "rmse_m": round(float(np.sqrt((err ** 2).mean())), 3),
        "mae_m": round(float(np.abs(err).mean()), 3),
        "bias_m": round(float(err.mean()), 3),
        "ref_mean_m": round(float(r.mean()), 3),
        "pred_mean_m": round(float(p.mean()), 3),
    }
    if r.std() > 1e-6 and p.std() > 1e-6:
        out["correlation"] = round(float(np.corrcoef(p, r)[0, 1]), 4)
        denom = ((r - r.mean()) ** 2).sum()
        out["r2"] = round(float(1 - (err ** 2).sum() / denom), 4)
    return out


def run(scene: str, work_root: Path, laz_paths,
        building_only_thresh: float, metric: str = "eave") -> dict:
    work_dir = work_root / scene
    meta = json.loads((work_dir / "ingest.json").read_text())

    # The working grid's geotransform comes from the source raster, scaled by
    # whatever downsampling ingest applied.
    with rasterio.open(meta["source_path"]) as src:
        base = src.transform
        f = meta.get("downsample_factor", 1.0)
        transform = rasterio.Affine(base.a * f, base.b, base.c,
                                    base.d, base.e * f, base.f)
    shape = (meta["height"], meta["width"])

    # LiDAR's highest return is the roof RIDGE; shadow geometry measures the
    # EAVE. Compare like with like or the difference appears as a constant
    # negative bias that is not actually an error.
    pred_name = "ndsm_ridge_m.npy" if metric == "ridge" else "ndsm_m.npy"
    pred_path = work_dir / pred_name
    if not pred_path.exists():
        raise SystemExit(f"error: {pred_path} not found - re-run pipeline.planb")
    print(f"  comparing: {metric} height ({pred_name})")
    pred = np.load(pred_path).astype(np.float32)
    if pred.shape != shape:
        raise SystemExit(f"error: ndsm_m.npy is {pred.shape}, expected {shape}")

    x, y, z, cls = load_points(laz_paths, meta["crs"])

    dsm, n_in = rasterise(x, y, z, transform, shape, "max")
    if dsm is None or n_in < 1000:
        raise SystemExit(
            f"\nerror: only {n_in} points fall inside the scene footprint.\n"
            f"  The LiDAR tile and the imagery tile do not overlap.\n"
            f"  Scene centre is {meta['center_lat']:.5f}, {meta['center_lon']:.5f}\n"
            f"  — download a point cloud for that exact bounding box."
        )
    print(f"  {n_in:,} points inside the scene ({100 * n_in / x.size:.1f}%)")

    ground = cls == GROUND_CLASS
    print(f"  ground-classified returns: {ground.sum():,} "
          f"({100 * ground.mean():.1f}%)")
    if ground.sum() < 500:
        raise SystemExit(
            "\nerror: too few ground-classified points to build a DTM.\n"
            "  This cloud is unclassified. Pick a 3DEP dataset whose\n"
            "  description mentions classified returns."
        )

    dtm, _ = rasterise(x[ground], y[ground], z[ground], transform, shape, "min")

    coverage = float(np.isfinite(dsm).mean())
    print(f"  DSM cell coverage: {coverage * 100:.1f}%")

    dsm, dtm = fill_gaps(dsm), fill_gaps(dtm)
    # Smooth the terrain surface: ground returns are sparse under buildings,
    # so the raw DTM is noisy exactly where it matters most.
    dtm = ndimage.uniform_filter(dtm, size=15, mode="nearest")

    ref = np.maximum(dsm - dtm, 0.0).astype(np.float32)
    np.save(work_dir / "ndsm_ref.npy", ref)

    valid = np.isfinite(ref) & np.isfinite(pred)
    all_m = metrics(pred, ref, valid)

    # Buildings only. The all-pixel figure is flattered by the vast area of
    # street and open ground where both maps correctly read zero -- report it,
    # but lead with the number that is actually about buildings.
    bmask = valid & ((ref > building_only_thresh) | (pred > building_only_thresh))
    bld_m = metrics(pred, ref, bmask)

    per_building = {}
    labels_path = work_dir / "labels.npy"
    if labels_path.exists():
        labels = np.load(labels_path)
        pairs = []
        for lid in range(1, int(labels.max()) + 1):
            sel = labels == lid
            if sel.sum() < 20:
                continue
            pairs.append((float(np.percentile(pred[sel], 75)),
                          float(np.percentile(ref[sel], 75))))
        if len(pairs) >= 5:
            p = np.array([a for a, _ in pairs])
            r = np.array([b for _, b in pairs])
            e = p - r
            per_building = {
                "n_buildings": len(pairs),
                "rmse_m": round(float(np.sqrt((e ** 2).mean())), 3),
                "mae_m": round(float(np.abs(e).mean()), 3),
                "bias_m": round(float(e.mean()), 3),
                "correlation": round(float(np.corrcoef(p, r)[0, 1]), 4),
            }

    report = {
        "scene": scene,
        "metric": metric,
        "lidar_source": ", ".join(p.name for p in laz_paths),
        "dsm_coverage": round(coverage, 4),
        "all_pixels": all_m,
        "building_pixels": bld_m,
        "per_building": per_building,
    }
    (work_dir / "validation.json").write_text(json.dumps(report, indent=2))

    print("\n  ALL PIXELS")
    for k in ("rmse_m", "mae_m", "bias_m", "correlation"):
        if k in all_m:
            print(f"    {k:<12}: {all_m[k]}")
    print("\n  BUILDING PIXELS  (lead with this)")
    for k in ("rmse_m", "mae_m", "bias_m", "correlation"):
        if k in bld_m:
            print(f"    {k:<12}: {bld_m[k]}")
    if per_building:
        print(f"\n  PER BUILDING  (n={per_building['n_buildings']})")
        for k in ("rmse_m", "mae_m", "bias_m", "correlation"):
            print(f"    {k:<12}: {per_building[k]}")

    print(f"\n  wrote {work_dir / 'ndsm_ref.npy'}  (LiDAR reference nDSM)")
    print(f"  wrote {work_dir / 'validation.json'}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--lidar", type=Path, nargs="+", required=True,
                    help="one or more .las/.laz files")
    ap.add_argument("--work", type=Path, default=Path("data/work"))
    ap.add_argument("--metric", choices=["eave", "ridge"], default="eave",
                    help="which surface to compare against LiDAR")
    ap.add_argument("--building-threshold", type=float, default=2.0,
                    help="metres above ground that counts as a building")
    args = ap.parse_args()

    for p in args.lidar:
        if not p.exists():
            raise SystemExit(f"error: {p} not found")
    print(f"\n  scene: {args.scene}")
    run(args.scene, args.work, args.lidar, args.building_threshold,
        args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())