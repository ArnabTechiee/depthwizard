"""
DepthWizard - Plan B: direct shadow-derived metric DSM (brief section 5).

Zero-shot monocular depth failed to track building height in a dense urban
core: fitting a global scale factor against shadow anchors gave a negative
held-out R2, meaning the depth value for a building carries no usable
information about how tall it is.

So we stop using it as a height source. Each building's height comes from its
OWN cast shadow -- a direct physical measurement, not a learned prior:

    h = L_pixels x GSD / tan(sun_elevation)

Depth still earns its place: it supplies the ground surface and the footprint
segmentation. It just no longer decides the metres.

Validation here is repeatability, not a fitted holdout: each building's rays
are split in half at random and a height computed from each half. The spread
between halves is an honest measure of how stable the measurement is, and it
needs no external ground truth.

    python -m pipeline.planb antakya
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from pipeline.calibrate import shadow_mask, building_mask, shadow_direction


def split_blocks(mask: np.ndarray, ndsm: np.ndarray, min_area_px: int) -> np.ndarray:
    """
    Break merged city blocks into individual buildings.

    A raw threshold on the nDSM fuses a whole terrace into one component, and
    a fused component is useless: its shadow is measured at the block's outer
    edge while its area spans dozens of buildings. Watershed on the smoothed
    nDSM separates them at the roof-height saddles between neighbours.
    """
    try:
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
    except ImportError:
        print("  ! scikit-image missing, using unsplit components")
        return ndimage.label(mask)[0]

    smooth = ndimage.gaussian_filter(ndsm * mask, 2.0)
    footprint_px = max(5, int(math.sqrt(min_area_px) * 0.8) | 1)

    peaks = peak_local_max(smooth, labels=mask, min_distance=footprint_px,
                           exclude_border=False)
    if peaks.shape[0] < 2:
        return ndimage.label(mask)[0]

    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = i

    labels = watershed(-smooth, markers, mask=mask)

    sizes = ndimage.sum(mask, labels, range(1, labels.max() + 1))
    too_small = np.nonzero(sizes < min_area_px)[0] + 1
    labels[np.isin(labels, too_small)] = 0
    return labels


def ray_lengths(labels, label_id, shadows, buildings, d_row, d_col, max_len):
    """All valid shadow-ray lengths for one building, in pixels."""
    h, w = labels.shape
    ys, xs = np.nonzero(labels == label_id)
    if ys.size == 0:
        return np.array([])

    edge = []
    for y, x in zip(ys, xs):
        ny, nx = int(round(y + d_row)), int(round(x + d_col))
        if not (0 <= ny < h and 0 <= nx < w) or labels[ny, nx] != label_id:
            edge.append((y, x))
    if len(edge) < 3:
        return np.array([])
    if len(edge) > 60:
        edge = [edge[i] for i in np.linspace(0, len(edge) - 1, 60).astype(int)]

    out = []
    for y, x in edge:
        start = 0
        for step in range(1, 25):
            ny, nx = int(round(y + d_row * step)), int(round(x + d_col * step))
            if not (0 <= ny < h and 0 <= nx < w):
                break
            if not buildings[ny, nx]:
                start = step
                break
        if start == 0:
            continue

        length = hits = 0
        blocked = False
        for step in range(start, max_len + 1):
            ny, nx = int(round(y + d_row * step)), int(round(x + d_col * step))
            if not (0 <= ny < h and 0 <= nx < w):
                blocked = True
                break
            if buildings[ny, nx]:
                blocked = True
                break
            if shadows[ny, nx]:
                length = step - start + 1
                hits += 1
            elif step - start + 1 > length + 1:
                break
        if not blocked and length > 0 and hits / float(length) >= 0.8:
            out.append(length)
    return np.array(out, dtype=np.float32)


def run(scene: str, work_root: Path, min_area_m2: float,
        default_height_m: float, roof_pitch_deg: float = 22.6,
        max_rise_m: float = 4.0) -> dict:
    work_dir = work_root / scene
    meta = json.loads((work_dir / "ingest.json").read_text())
    if meta["mode"] != "absolute":
        raise SystemExit(f"error: {scene} is RELATIVE - cannot produce metres.")

    gsd = meta["gsd_m"]
    tan_elev = math.tan(math.radians(meta["sun_elevation_deg"]))
    d_row, d_col = shadow_direction(meta["sun_azimuth_deg"])
    max_len = int(200.0 / tan_elev / gsd)
    tan_pitch = math.tan(math.radians(roof_pitch_deg))

    ndsm_rel = np.load(work_dir / "ndsm_rel.npy").astype(np.float32)
    rgb = np.load(work_dir / "rgb.npy")

    min_area_px = int(min_area_m2 / (gsd ** 2))
    shadows = shadow_mask(rgb)
    buildings = building_mask(ndsm_rel, min_area_px)
    labels = split_blocks(buildings, ndsm_rel, min_area_px)
    n = int(labels.max())

    print(f"\n  GSD {gsd:.3f} m/px   sun {meta['sun_elevation_deg']:.1f} deg")
    print(f"  buildings after splitting : {n}")

    rng = np.random.default_rng(0)
    records, split_errs = [], []

    for label_id in range(1, n + 1):
        lens = ray_lengths(labels, label_id, shadows, buildings,
                           d_row, d_col, max_len)
        area_px = int((labels == label_id).sum())
        if area_px == 0:
            continue

        ys, xs = np.nonzero(labels == label_id)
        rec = {"id": label_id,
               "row": float(ys.mean()), "col": float(xs.mean()),
               "area_m2": round(area_px * gsd * gsd, 1)}

        if lens.size >= 4:
            height = float(np.median(lens)) * gsd / tan_elev
            rec.update(height_m=round(height, 2), n_rays=int(lens.size),
                       method="shadow")

            # Shadow geometry measures to the EAVE: the shadow is cast by the
            # roof's outer edge, not its ridge. LiDAR reports the highest
            # return, which on a pitched roof is the ridge. The two are both
            # correct and differ by the roof's rise -- so we report both
            # rather than treating the gap as error.
            #
            # Rise is derived from a stated architectural prior (a typical
            # residential pitch) applied to the building's own short span,
            # NOT fitted to the reference data.
            rows_i, cols_i = np.nonzero(labels == label_id)
            span_px = min(int(rows_i.max() - rows_i.min()) + 1,
                          int(cols_i.max() - cols_i.min()) + 1)
            rise = 0.5 * span_px * gsd * tan_pitch
            rise = min(rise, max_rise_m)
            rec["ridge_m"] = round(height + rise, 2)
            rec["roof_rise_m"] = round(rise, 2)

            perm = rng.permutation(lens.size)
            a = lens[perm[: lens.size // 2]]
            b = lens[perm[lens.size // 2:]]
            if a.size >= 2 and b.size >= 2:
                ha = float(np.median(a)) * gsd / tan_elev
                hb = float(np.median(b)) * gsd / tan_elev
                rec["repeatability_m"] = round(abs(ha - hb), 2)
                split_errs.append(abs(ha - hb))
        else:
            rec.update(height_m=None, n_rays=int(lens.size), method="none")
        records.append(rec)

    measured = [r for r in records if r["method"] == "shadow"]
    print(f"  measured from shadow      : {len(measured)} / {len(records)}")

    if not measured:
        raise SystemExit("  no buildings measurable - check the shadow mask")

    heights = np.array([r["height_m"] for r in measured])
    fallback = float(np.median(heights))
    print(f"\n  heights (m): min {heights.min():.1f}  median "
          f"{np.median(heights):.1f}  max {heights.max():.1f}")
    print(f"  floors approx: {np.median(heights) / 3.0:.1f} storeys median")

    # Rasterise: each footprint gets its own measured height.
    dsm = np.zeros_like(ndsm_rel, dtype=np.float32)
    for rec in records:
        h = rec["height_m"]
        if h is None:
            h = default_height_m or fallback
            rec["height_m"] = round(h, 2)
            rec["method"] = "median-fill"
        dsm[labels == rec["id"]] = h

    ridge = np.zeros_like(ndsm_rel, dtype=np.float32)
    for rec in records:
        ridge[labels == rec["id"]] = rec.get("ridge_m", rec["height_m"])

    dsm = ndimage.gaussian_filter(dsm, 0.8)     # soften raster stair-steps
    ridge = ndimage.gaussian_filter(ridge, 0.8)

    result = {
        "scene": scene,
        "method": "shadow-direct (Plan B)",
        "n_buildings": len(records),
        "n_measured": len(measured),
        "median_height_m": round(float(np.median(heights)), 2),
        "max_height_m": round(float(heights.max()), 2),
        "fallback_height_m": round(fallback, 2),
        "roof_pitch_deg": roof_pitch_deg,
        "median_roof_rise_m": round(float(np.median(
            [r["roof_rise_m"] for r in measured if "roof_rise_m" in r] or [0])), 2),
    }
    if split_errs:
        se = np.array(split_errs)
        result["repeatability_mae_m"] = round(float(se.mean()), 2)
        result["repeatability_rmse_m"] = round(float(np.sqrt((se ** 2).mean())), 2)
        print(f"\n  REPEATABILITY (split-half, no ground truth needed)")
        print(f"    MAE  : {result['repeatability_mae_m']:.2f} m")
        print(f"    RMSE : {result['repeatability_rmse_m']:.2f} m")

    np.save(work_dir / "ndsm_m.npy", dsm)
    np.save(work_dir / "ndsm_ridge_m.npy", ridge)
    np.save(work_dir / "labels.npy", labels.astype(np.int32))
    json.dump({**result, "buildings": records},
              open(work_dir / "buildings.json", "w"), indent=2)

    print(f"\n  wrote {work_dir / 'ndsm_m.npy'}   (metric heights)")
    print(f"  wrote {work_dir / 'buildings.json'} ({len(records)} rows)")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--work", type=Path, default=Path("data/work"))
    ap.add_argument("--min-area", type=float, default=120.0)
    ap.add_argument("--default-height", type=float, default=0.0,
                    help="height for unmeasurable buildings (0 = use median)")
    ap.add_argument("--roof-pitch-deg", type=float, default=22.6,
                    help="assumed roof pitch for the eave->ridge estimate "
                         "(22.6 deg = a 5:12 residential pitch)")
    ap.add_argument("--max-rise", type=float, default=4.0,
                    help="cap on the eave->ridge rise, metres")
    args = ap.parse_args()
    run(args.scene, args.work, args.min_area, args.default_height,
        args.roof_pitch_deg, args.max_rise)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
