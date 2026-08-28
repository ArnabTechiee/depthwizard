"""
DepthWizard - Stage 3a: ground surface extraction (T0-4).

Monocular depth models trained on ground-level photography impose a receding
ground plane on nadir imagery, producing a large low-frequency ramp that has
nothing to do with terrain. On a real scene that ramp can be several times
larger than the building heights you actually want.

Decompose it:

    depth  =  ground surface  +  nDSM

The ground surface is estimated by grey-scale morphological opening with a
structuring element WIDER than the largest building footprint. Opening keeps
whatever survives an erode-then-dilate, so anything narrower than the element
-- every building -- is erased, and what remains is terrain plus the model's
fabricated ramp. Subtract it and the buildings are left standing on a flat
baseline.

The element size is the one parameter that matters. Too small and it climbs
onto rooftops, flattening the buildings you are trying to measure. Too large
and real topography leaks into the nDSM.

    python -m pipeline.ground synthetic
    python -m pipeline.ground antakya --max-building-m 80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import grey_opening, uniform_filter


def _resize_nearest(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.linspace(0, arr.shape[0] - 1, shape[0]).round().astype(int)
    cols = np.linspace(0, arr.shape[1] - 1, shape[1]).round().astype(int)
    return arr[np.ix_(rows, cols)]


def estimate_ground(depth: np.ndarray, element_px: int) -> np.ndarray:
    """Morphological opening, then smoothed to remove blocky artefacts."""
    element_px = max(3, int(element_px) | 1)          # odd, >= 3
    opened = grey_opening(depth, size=(element_px, element_px), mode="nearest")
    return uniform_filter(opened, size=element_px, mode="nearest")


def run(scene: str, work_root: Path, max_building_m: float,
        truth_path: Path | None) -> dict:
    work_dir = work_root / scene

    depth = np.load(work_dir / "depth_norm.npy").astype(np.float32)
    meta = json.loads((work_dir / "ingest.json").read_text())

    gsd = meta.get("gsd_m")
    if gsd:
        element_px = int(round(max_building_m / gsd))
        print(f"  GSD {gsd:.3f} m/px -> structuring element {element_px} px "
              f"({max_building_m:.0f} m)")
    else:
        element_px = max(31, min(depth.shape) // 12)
        print(f"  no GSD - falling back to {element_px} px element")

    if element_px >= min(depth.shape) // 2:
        element_px = min(depth.shape) // 4
        print(f"  ! element too large for image, clamped to {element_px} px")

    ground = estimate_ground(depth, element_px)
    ndsm = depth - ground
    ndsm = np.maximum(ndsm, 0.0)          # nothing sits below ground

    print(f"\n  raw depth  range: [{depth.min():.3f}, {depth.max():.3f}]  "
          f"spread {depth.max() - depth.min():.3f}")
    print(f"  ground     range: [{ground.min():.3f}, {ground.max():.3f}]  "
          f"spread {ground.max() - ground.min():.3f}   <- the fabricated ramp")
    print(f"  nDSM       range: [{ndsm.min():.3f}, {ndsm.max():.3f}]  "
          f"spread {ndsm.max() - ndsm.min():.3f}")

    report = {"scene": scene, "element_px": element_px,
              "max_building_m": max_building_m}

    if truth_path is not None and truth_path.exists():
        truth = np.load(truth_path).astype(np.float32)
        if truth.shape != depth.shape:
            truth = _resize_nearest(truth, depth.shape)

        before = float(np.corrcoef(depth.ravel(), truth.ravel())[0, 1])
        after = float(np.corrcoef(ndsm.ravel(), truth.ravel())[0, 1])

        print(f"\n  correlation vs truth")
        print(f"    before detrend : {before:+.4f}")
        print(f"    after  detrend : {after:+.4f}   "
              f"({'+' if after > before else ''}{(after - before):.4f})")

        is_building = truth > 0.5
        if is_building.any() and (~is_building).any():
            spread = float(ndsm.std()) or 1.0
            sep = float(ndsm[is_building].mean() - ndsm[~is_building].mean()) / spread
            print(f"    separation     : {sep:+.2f} sigma")
            report["separation_sigma"] = sep

        report["correlation_before"] = before
        report["correlation_after"] = after

        if after > 0.6:
            print("\n  -> Strong. The ramp was the problem; zero-shot is fine.")
        elif after > 0.35:
            print("\n  -> Workable. Proceed, but shadow anchors carry the metric claim.")
        else:
            print("\n  -> Weak. If this repeats on real imagery, go to Plan B:")
            print("     footprint segmentation + shadow heights, no depth model.")

    np.save(work_dir / "ndsm_rel.npy", ndsm.astype(np.float32))
    np.save(work_dir / "ground_rel.npy", ground.astype(np.float32))
    with open(work_dir / "ground.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n  wrote {work_dir / 'ndsm_rel.npy'}  (relative height above ground)")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--work", type=Path, default=Path("data/work"))
    ap.add_argument("--max-building-m", type=float, default=60.0,
                    help="widest building footprint expected, in metres")
    ap.add_argument("--truth", type=Path, default=None)
    args = ap.parse_args()

    truth = args.truth
    if truth is None:
        guess = Path("data/raw") / f"{args.scene}.truth.npy"
        truth = guess if guess.exists() else None
        if truth:
            print(f"using truth: {truth}")

    print(f"\n  scene: {args.scene}")
    run(args.scene, args.work, args.max_building_m, truth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
