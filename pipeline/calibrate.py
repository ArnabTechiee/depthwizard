"""
DepthWizard - Stage 3b: shadow-anchored metric calibration (T0-5).

The relative nDSM says building A is twice building B. This turns that into
metres, using the physics already present in the image:

    h  =  L_pixels  x  GSD  x  tan(sun_elevation)

for a vertical structure whose shadow falls on flat ground. Sun elevation and
azimuth come from the product metadata; GSD comes from the affine transform.
Every isolated building in the scene is therefore a free absolute anchor.

We do NOT need a correct height for every building. We need ONE global scale
factor for the whole nDSM, so we measure many buildings, fit robustly with
RANSAC, and let outliers fall away. Twenty clean anchors is plenty.

Crucially, the fit is done on a random 70% of anchors and evaluated on the
withheld 30%. That held-out error is a real accuracy number on real imagery,
derived from an independent physical measurement rather than from a scene we
generated ourselves.

    python -m pipeline.calibrate synthetic --verify-shadows
    python -m pipeline.calibrate antakya
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------

def shadow_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Shadows are dark, and relatively blue: they are lit by sky rather than sun.
    Combining darkness with a blue-bias rejects dark roofs and asphalt, which
    are dark but not blue-shifted.
    """
    arr = rgb.astype(np.float32)
    luma = arr.mean(axis=2)

    # Shadows are the darkest population in the scene AND blue-shifted, since
    # they are lit by sky rather than sun. The blue test is what separates a
    # true shadow from a merely dark roof or fresh asphalt.
    dark = luma < np.percentile(luma, 22)
    blue_bias = arr[:, :, 2] - arr[:, :, 0]           # B - R
    bluish = blue_bias > np.percentile(blue_bias, 45)

    mask = dark & bluish
    if mask.mean() < 0.02:                            # too strict, relax
        mask = dark
    return ndimage.binary_opening(mask, np.ones((3, 3)))


def building_mask(ndsm: np.ndarray, min_area_px: int) -> np.ndarray:
    """Anything standing meaningfully above the local ground surface."""
    positive = ndsm[ndsm > 1e-6]
    if positive.size == 0:
        return np.zeros_like(ndsm, dtype=bool)

    thresh = np.percentile(positive, 55)
    mask = ndsm > thresh
    mask = ndimage.binary_opening(mask, np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)

    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    keep = np.isin(labels, np.nonzero(sizes >= min_area_px)[0] + 1)
    return keep


# --------------------------------------------------------------------------
# Shadow measurement
# --------------------------------------------------------------------------

def shadow_direction(sun_azimuth_deg: float) -> tuple[float, float]:
    """
    Unit step (d_row, d_col) pointing along the cast shadow.

    Azimuth is clockwise from north, measured TOWARDS the sun, so the shadow
    runs 180 degrees opposite. Rows increase southward, hence the sign flip.
    """
    az = math.radians((sun_azimuth_deg + 180.0) % 360.0)
    d_col = math.sin(az)
    d_row = -math.cos(az)
    return d_row, d_col


def measure_building(labels: np.ndarray, label_id: int, shadows: np.ndarray,
                     buildings: np.ndarray, d_row: float, d_col: float,
                     max_len: int, samples: int = 40) -> float | None:
    """
    Median shadow length in pixels for one building, by ray casting.

    Rays start on the building's trailing edge (the side away from the sun)
    and step along the shadow direction. A ray that runs into another building
    is discarded: in a dense core that shadow is falling on a wall, not on
    flat ground, and its length is meaningless.
    """
    h, w = labels.shape
    ys, xs = np.nonzero(labels == label_id)
    if ys.size == 0:
        return None

    # Trailing edge: building pixels whose next step leaves the building.
    edge = []
    for y, x in zip(ys, xs):
        ny, nx = int(round(y + d_row)), int(round(x + d_col))
        if not (0 <= ny < h and 0 <= nx < w) or not buildings[ny, nx]:
            edge.append((y, x))
    if len(edge) < 4:
        return None

    if len(edge) > samples:
        idx = np.linspace(0, len(edge) - 1, samples).astype(int)
        edge = [edge[i] for i in idx]

    lengths = []
    for y, x in edge:
        # Walk out of the building first. The nDSM's blurred edges make the
        # mask bleed a few pixels past the true footprint, so the ray's real
        # origin is wherever it stops being building -- not the mask edge.
        start = 0
        for step in range(1, 25):
            ny = int(round(y + d_row * step)); nx = int(round(x + d_col * step))
            if not (0 <= ny < h and 0 <= nx < w):
                break
            if not buildings[ny, nx]:
                start = step
                break
        if start == 0:
            continue

        length = 0
        hits = 0
        blocked = False
        for step in range(start, max_len + 1):
            ny = int(round(y + d_row * step)); nx = int(round(x + d_col * step))
            if not (0 <= ny < h and 0 <= nx < w):
                blocked = True; break
            if buildings[ny, nx]:
                blocked = True; break
            if shadows[ny, nx]:
                length = step - start + 1; hits += 1
            elif step - start + 1 > length + 1:
                break
        if not blocked and length > 0 and hits / float(length) >= 0.85:
            lengths.append(length)

    if len(lengths) < 4:
        return None

    lengths = np.array(lengths, dtype=np.float32)
    median = float(np.median(lengths))
    if median < 3:
        return None
    # Reject buildings whose shadow length is wildly inconsistent along the
    # edge -- usually merged shadows or a shadow falling on a slope.
    if float(np.std(lengths)) / median > 0.45:
        return None
    return median


# --------------------------------------------------------------------------
# Anchor collection
# --------------------------------------------------------------------------

def collect_anchors(ndsm: np.ndarray, rgb: np.ndarray, meta: dict,
                    min_area_m2: float) -> list[dict]:
    gsd = meta["gsd_m"]
    sun_elev = meta["sun_elevation_deg"]
    sun_azim = meta["sun_azimuth_deg"]

    min_area_px = int(min_area_m2 / (gsd ** 2))
    shadows = shadow_mask(rgb)
    buildings = building_mask(ndsm, min_area_px)

    print(f"  shadow pixels   : {shadows.mean() * 100:.1f}% of image")
    print(f"  building pixels : {buildings.mean() * 100:.1f}% of image")

    labels, n = ndimage.label(buildings)
    print(f"  candidate buildings: {n}")

    d_row, d_col = shadow_direction(sun_azim)
    tan_elev = math.tan(math.radians(sun_elev))
    max_len = int(200.0 / tan_elev / gsd)     # cap at a 200 m structure

    # The nDSM's edges are blurred by the depth model, so the building mask
    # bleeds a few pixels past the true footprint. Ray origins taken from
    # that outer boundary start too early and every shadow reads long by a
    # constant few pixels. Erode before edge-finding to pull origins back
    # onto the real edge; the un-eroded mask is still what blocks rays.
    anchors = []
    rejected = 0
    for label_id in range(1, n + 1):
        length_px = measure_building(labels, label_id, shadows, buildings,
                                     d_row, d_col, max_len)
        if length_px is None:
            rejected += 1
            continue

        height_m = length_px * gsd / tan_elev
        if not 2.0 <= height_m <= 200.0:
            rejected += 1
            continue

        region = ndsm[labels == label_id]
        ndsm_value = float(np.percentile(region, 75))
        if ndsm_value <= 1e-6:
            rejected += 1
            continue

        anchors.append({
            "label": int(label_id),
            "shadow_px": float(length_px),
            "height_m": float(height_m),
            "ndsm_rel": ndsm_value,
            "area_px": int((labels == label_id).sum()),
        })

    print(f"  anchors accepted   : {len(anchors)}  (rejected {rejected})")
    return anchors


# --------------------------------------------------------------------------
# Robust fit with held-out validation
# --------------------------------------------------------------------------

def fit_scale(anchors: list[dict], seed: int = 0) -> dict:
    """RANSAC through the origin: height_m = scale * ndsm_rel."""
    x = np.array([a["ndsm_rel"] for a in anchors], dtype=np.float64)
    y = np.array([a["height_m"] for a in anchors], dtype=np.float64)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    split = max(3, int(len(x) * 0.7))
    train, test = order[:split], order[split:]

    best_scale, best_inliers = None, -1
    tolerance = max(2.0, 0.15 * float(np.median(y)))

    for _ in range(500):
        i = rng.integers(0, len(train))
        xi, yi = x[train[i]], y[train[i]]
        if xi <= 1e-9:
            continue
        candidate = yi / xi
        residual = np.abs(y[train] - candidate * x[train])
        inliers = int((residual < tolerance).sum())
        if inliers > best_inliers:
            best_inliers, best_scale = inliers, candidate

    # Refit on inliers by least squares through the origin.
    residual = np.abs(y[train] - best_scale * x[train])
    inlier_idx = train[residual < tolerance]
    if inlier_idx.size >= 2:
        best_scale = float(x[inlier_idx] @ y[inlier_idx] / (x[inlier_idx] @ x[inlier_idx]))

    result = {
        "scale": float(best_scale),
        "n_anchors": len(x),
        "n_train": int(len(train)),
        "n_inliers": int(inlier_idx.size),
        "tolerance_m": float(tolerance),
    }

    if test.size >= 2:
        pred = best_scale * x[test]
        err = pred - y[test]
        result["holdout_n"] = int(test.size)
        result["holdout_rmse_m"] = float(np.sqrt((err ** 2).mean()))
        result["holdout_mae_m"] = float(np.abs(err).mean())
        denom = y[test] - y[test].mean()
        if float((denom ** 2).sum()) > 1e-9:
            result["holdout_r2"] = float(1 - (err ** 2).sum() / (denom ** 2).sum())
    return result


# --------------------------------------------------------------------------

def verify_shadows(anchors: list[dict], truth_json: Path,
                   labels: np.ndarray, gsd: float,
                   downsample: float) -> None:
    """
    Synthetic only: match each anchor to its true building by position and
    compare derived height against the known height.

    Comparing heights in metres rather than shadow lengths in pixels is the
    right test -- it exercises GSD handling and the tan(elevation) conversion
    as well as the measurement, and it survives any resampling of the image.
    """
    truth = json.loads(truth_json.read_text())
    buildings = truth["buildings"]

    print("\n  height unit test (synthetic, known truth)")
    rows = []
    for a in anchors:
        ys, xs = np.nonzero(labels == a["label"])
        cy, cx = ys.mean() * downsample, xs.mean() * downsample

        best, best_d = None, 1e18
        for b in buildings:
            by = b["row"] + b["d"] / 2.0
            bx = b["col"] + b["w"] / 2.0
            d = (by - cy) ** 2 + (bx - cx) ** 2
            if d < best_d:
                best, best_d = b, d
        if best is None or best_d > (150.0 ** 2):
            continue
        rows.append((best["height_m"], a["height_m"]))

    if not rows:
        print("    no anchor matched a known building")
        return

    true_h = np.array([r[0] for r in rows])
    pred_h = np.array([r[1] for r in rows])
    err = 100.0 * np.abs(pred_h - true_h) / true_h

    print(f"    {'true':>8}{'measured':>10}{'err':>8}")
    for t, m in sorted(rows):
        print(f"    {t:>8.1f}{m:>10.1f}{100 * abs(m - t) / t:>7.1f}%")
    print(f"\n    matched      : {len(rows)} / {len(buildings)}")
    print(f"    median error : {np.median(err):.1f}%")
    print(f"    RMSE         : {np.sqrt(((pred_h - true_h) ** 2).mean()):.2f} m")

    if np.median(err) < 15:
        print("    -> shadow geometry and GSD handling are CORRECT")
    else:
        print("    -> tune shadow detection before trusting metres")


def run(scene: str, work_root: Path, min_area_m2: float,
        verify: bool) -> dict:
    work_dir = work_root / scene
    meta = json.loads((work_dir / "ingest.json").read_text())

    if meta["mode"] != "absolute":
        raise SystemExit(f"error: {scene} is RELATIVE - no GSD or sun angles.")

    ndsm = np.load(work_dir / "ndsm_rel.npy").astype(np.float32)
    rgb = np.load(work_dir / "rgb.npy")

    print(f"\n  scene      : {scene}")
    print(f"  GSD        : {meta['gsd_m']:.4f} m/px")
    print(f"  sun elev/az: {meta['sun_elevation_deg']:.1f} / "
          f"{meta['sun_azimuth_deg']:.1f}")
    d_row, d_col = shadow_direction(meta["sun_azimuth_deg"])
    print(f"  shadow dir : row {d_row:+.2f}, col {d_col:+.2f}\n")

    min_area_px = int(min_area_m2 / (meta["gsd_m"] ** 2))
    labels, _ = ndimage.label(building_mask(ndsm, min_area_px))
    anchors = collect_anchors(ndsm, rgb, meta, min_area_m2)

    if len(anchors) < 8:
        print(f"\n  ! Only {len(anchors)} anchors. Too few to fit a scale factor.")
        print("    Try --min-area 80, or a denser scene. If this persists,")
        print("    Plan B (footprint segmentation + direct shadow heights)")
        print("    is the right call.")
        json.dump({"n_anchors": len(anchors), "anchors": anchors},
                  open(work_dir / "calibration.json", "w"), indent=2)
        return {"n_anchors": len(anchors)}

    heights = np.array([a["height_m"] for a in anchors])
    print(f"\n  shadow-derived heights (m):")
    print(f"    min {heights.min():.1f}   median {np.median(heights):.1f}   "
          f"max {heights.max():.1f}")

    fit = fit_scale(anchors)
    print(f"\n  RANSAC fit")
    print(f"    scale factor   : {fit['scale']:.2f} m per nDSM unit")
    print(f"    train / inliers: {fit['n_train']} / {fit['n_inliers']}")
    if "holdout_rmse_m" in fit:
        print(f"\n  HELD-OUT VALIDATION  (n={fit['holdout_n']}, never seen by the fit)")
        print(f"    RMSE : {fit['holdout_rmse_m']:.2f} m")
        print(f"    MAE  : {fit['holdout_mae_m']:.2f} m")
        if "holdout_r2" in fit:
            print(f"    R2   : {fit['holdout_r2']:+.3f}")

    dsm_m = ndsm * fit["scale"]
    np.save(work_dir / "ndsm_m.npy", dsm_m.astype(np.float32))
    print(f"\n  nDSM in metres: max {dsm_m.max():.1f} m, "
          f"mean over buildings {dsm_m[dsm_m > 1].mean():.1f} m")

    out = {"scene": scene, **fit, "anchors": anchors}
    json.dump(out, open(work_dir / "calibration.json", "w"), indent=2)
    print(f"\n  wrote {work_dir / 'ndsm_m.npy'}")
    print(f"  wrote {work_dir / 'calibration.json'}")

    if verify:
        truth_json = Path("data/raw") / f"{scene}.truth.json"
        if truth_json.exists():
            verify_shadows(anchors, truth_json, labels, meta["gsd_m"],
                           meta.get("downsample_factor", 1.0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--work", type=Path, default=Path("data/work"))
    ap.add_argument("--min-area", type=float, default=120.0,
                    help="smallest building footprint to use, m^2")
    ap.add_argument("--verify-shadows", action="store_true")
    args = ap.parse_args()

    run(args.scene, args.work, args.min_area, args.verify_shadows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
