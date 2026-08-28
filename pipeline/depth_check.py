"""
DepthWizard - Stage 2b: verify and normalise the depth map from Colab.

Depth Anything V2 emits *affine-invariant inverse depth*: unitless, and only
defined up to an unknown scale and shift. Two things must be settled before
Stage 3 can calibrate it into metres, and both are silent failures if wrong.

1. ORIENTATION. Higher raw value means nearer the sensor. From nadir, nearer
   means taller -- so higher should mean taller. But this is a convention,
   not a guarantee, and if it is inverted your buildings become pits. Every
   downstream number stays plausible-looking while being exactly backwards.

   We settle it empirically on the synthetic scene, where true heights are
   known, and then apply that same convention to every real scene.

2. QUALITY. Correlation against known truth tells you whether zero-shot is
   good enough at all, which is the Section 4 backbone decision your brief
   says to make on measured evidence rather than assumption.

    python -m pipeline.depth_check synthetic
    python -m pipeline.depth_check morocco --orientation-from synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load(work_dir: Path, name: str) -> np.ndarray:
    path = work_dir / name
    if not path.exists():
        raise SystemExit(
            f"error: {path} not found.\n"
            f"       unzip the Colab output so it lands here as depth.npy"
        )
    return np.load(path)


def _resize_nearest(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Cheap nearest-neighbour resize, no OpenCV dependency."""
    rows = (np.linspace(0, arr.shape[0] - 1, shape[0])).round().astype(int)
    cols = (np.linspace(0, arr.shape[1] - 1, shape[1])).round().astype(int)
    return arr[np.ix_(rows, cols)]


def analyse(scene: str, work_root: Path, truth_path: Path | None,
            orientation: int | None) -> dict:
    work_dir = work_root / scene
    depth = _load(work_dir, "depth.npy").astype(np.float32)

    with open(work_dir / "ingest.json") as fh:
        meta = json.load(fh)

    print(f"\n  scene       : {scene}  ({meta['mode'].upper()})")
    print(f"  depth shape : {depth.shape[1]} x {depth.shape[0]}")
    print(f"  ingest shape: {meta['width']} x {meta['height']}")

    if depth.shape != (meta["height"], meta["width"]):
        print("  ! shape mismatch - resampling depth onto the ingest grid")
        depth = _resize_nearest(depth, (meta["height"], meta["width"]))

    finite = np.isfinite(depth)
    if not finite.all():
        print(f"  ! {(~finite).sum()} non-finite pixels, filling with median")
        depth[~finite] = float(np.median(depth[finite]))

    print(f"  raw range   : [{depth.min():.4f}, {depth.max():.4f}]")

    correlation = None
    if truth_path is not None:
        truth = np.load(truth_path).astype(np.float32)
        if truth.shape != depth.shape:
            truth = _resize_nearest(truth, depth.shape)

        correlation = float(np.corrcoef(depth.ravel(), truth.ravel())[0, 1])
        detected = 1 if correlation >= 0 else -1

        print(f"\n  correlation vs truth : {correlation:+.4f}")
        print(f"  detected orientation : {'+1 (higher = taller)' if detected > 0 else '-1 (INVERTED)'}")

        if abs(correlation) < 0.25:
            print("\n  ! Correlation is very weak. The depth map is not tracking")
            print("    height in any usable way. Check the RGB->BGR conversion")
            print("    in the notebook before blaming the model.")

        # Report separation between building pixels and ground pixels -- a more
        # honest signal than global correlation, which flat ground inflates.
        oriented = depth * detected
        is_building = truth > 0.5
        if is_building.any() and (~is_building).any():
            gap = float(oriented[is_building].mean() - oriented[~is_building].mean())
            spread = float(oriented.std()) or 1.0
            print(f"  building/ground separation : {gap / spread:+.2f} sigma")

        orientation = detected if orientation is None else orientation

    if orientation is None:
        print("\n  ! No truth and no --orientation given. Assuming +1.")
        print("    Verify against the synthetic scene before trusting this.")
        orientation = 1

    oriented = depth * orientation
    lo, hi = np.percentile(oriented, (1, 99))
    hi = hi if hi > lo else lo + 1e-6
    normalised = np.clip((oriented - lo) / (hi - lo), 0.0, 1.0)

    np.save(work_dir / "depth_norm.npy", normalised.astype(np.float32))

    report = {
        "scene": scene,
        "orientation": int(orientation),
        "raw_min": float(depth.min()),
        "raw_max": float(depth.max()),
        "p1": float(lo),
        "p99": float(hi),
        "correlation_vs_truth": correlation,
    }
    with open(work_dir / "depth.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n  wrote {work_dir / 'depth_norm.npy'}  (0=ground, 1=highest)")
    print(f"  wrote {work_dir / 'depth.json'}\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--work", type=Path, default=Path("data/work"))
    ap.add_argument("--truth", type=Path, default=None,
                    help="per-pixel true height .npy (synthetic scene only)")
    ap.add_argument("--orientation", type=int, choices=[1, -1], default=None,
                    help="force sign instead of detecting it")
    ap.add_argument("--orientation-from", type=str, default=None,
                    help="reuse the sign recorded for another scene")
    args = ap.parse_args()

    truth = args.truth
    if truth is None:
        guess = Path("data/raw") / f"{args.scene}.truth.npy"
        if guess.exists():
            truth = guess
            print(f"using truth: {guess}")

    orientation = args.orientation
    if orientation is None and args.orientation_from:
        ref = args.work / args.orientation_from / "depth.json"
        if not ref.exists():
            print(f"error: {ref} not found - run that scene first", file=sys.stderr)
            return 1
        orientation = json.loads(ref.read_text())["orientation"]
        print(f"reusing orientation {orientation:+d} from '{args.orientation_from}'")

    analyse(args.scene, args.work, truth, orientation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
