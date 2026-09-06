"""
DepthWizard - Stage 3: metric conversion.

Converts GAMUS relative depth into metric height using the
calibration learned during the 1000-scene GAMUS training run.

Current validated calibration:

    scale = 1.5554613
    shift = 0.22597118

The calibration is intentionally kept separate from GAMUS inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ============================================================
# VERIFIED GAMUS 1000-SCENE CALIBRATION
# ============================================================

DEFAULT_SCALE = 1.5554613
DEFAULT_SHIFT = 0.22597118


# ============================================================
# CONVERSION
# ============================================================

def convert_depth(
    depth: np.ndarray,
    scale: float,
    shift: float,
) -> np.ndarray:

    depth = depth.astype(np.float32)

    metric = (
        scale * depth +
        shift
    )

    return metric.astype(np.float32)


# ============================================================
# RUN
# ============================================================

def run(
    scene: str,
    work_root: Path,
    scale: float,
    shift: float,
) -> None:

    work_dir = work_root / scene

    depth_path = work_dir / "depth.npy"

    if not depth_path.exists():
        raise SystemExit(
            f"\nERROR: {depth_path} not found.\n"
            f"Run GAMUS inference first:\n\n"
            f"  python -m pipeline.depth_local {scene}\n"
        )

    print("=" * 70)
    print("DEPTHWIZARD — METRIC CONVERSION")
    print("=" * 70)

    print(f"Scene : {scene}")

    # --------------------------------------------------------
    # Load relative GAMUS depth
    # --------------------------------------------------------

    depth = np.load(depth_path).astype(np.float32)

    print("\nRelative GAMUS depth:")
    print("  shape :", depth.shape)
    print(
        "  range :",
        float(np.nanmin(depth)),
        "to",
        float(np.nanmax(depth)),
    )

    # --------------------------------------------------------
    # Apply calibration
    # --------------------------------------------------------

    metric = convert_depth(
        depth,
        scale,
        shift,
    )

    print("\nCalibration:")
    print("  scale :", scale)
    print("  shift :", shift)

    print("\nMetric height:")
    print(
        "  range :",
        float(np.nanmin(metric)),
        "to",
        float(np.nanmax(metric)),
    )

    print(
        "  mean  :",
        float(np.nanmean(metric)),
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = work_dir / "gamus_metric.npy"

    np.save(
        output_path,
        metric,
    )

    report = {
        "scene": scene,
        "model": "GAMUS 1000-scene SSI",
        "scale": float(scale),
        "shift": float(shift),
        "input": str(depth_path),
        "output": str(output_path),
        "input_min": float(np.nanmin(depth)),
        "input_max": float(np.nanmax(depth)),
        "metric_min": float(np.nanmin(metric)),
        "metric_max": float(np.nanmax(metric)),
    }

    report_path = work_dir / "metric.json"

    with open(
        report_path,
        "w",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print("\nSaved:")
    print(f"  {output_path}")
    print(f"  {report_path}")

    print("\n" + "=" * 70)
    print("METRIC CONVERSION COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    ap = argparse.ArgumentParser(
        description=(
            "Convert GAMUS relative depth "
            "to metric height"
        )
    )

    ap.add_argument(
        "scene",
    )

    ap.add_argument(
        "--work",
        type=Path,
        default=Path("data/work"),
    )

    ap.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
    )

    ap.add_argument(
        "--shift",
        type=float,
        default=DEFAULT_SHIFT,
    )

    args = ap.parse_args()

    run(
        args.scene,
        args.work,
        args.scale,
        args.shift,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())