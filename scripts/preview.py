"""
DepthWizard - quick look at anything in data/work/<scene>/.

You will want to eyeball arrays constantly from here on: the RGB crop, the
depth map, the nDSM, the shadow mask. This dumps any .npy to a PNG you can
open in VS Code.

    python scripts/preview.py Hurricane-Ian-9-26-2022_031331332011
    python scripts/preview.py synthetic --array depth_norm.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def to_image(arr: np.ndarray) -> Image.Image:
    arr = np.squeeze(arr)

    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr[:, :, :3])

    # Single channel -> percentile stretch to a turbo-ish grey ramp
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    lo, hi = np.percentile(arr[finite], (1, 99))
    hi = hi if hi > lo else lo + 1e-6
    norm = np.clip((arr - lo) / (hi - lo), 0, 1)
    return Image.fromarray((norm * 255).astype(np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--array", default="rgb.npy")
    ap.add_argument("--work", type=Path, default=Path("data/work"))
    args = ap.parse_args()

    src = args.work / args.scene / args.array
    if not src.exists():
        print(f"error: {src} not found")
        available = sorted(p.name for p in (args.work / args.scene).glob("*.npy"))
        if available:
            print("available:", ", ".join(available))
        return 1

    arr = np.load(src)
    img = to_image(arr)

    out = Path("outputs") / f"{args.scene}__{src.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)

    print(f"  {args.array}: shape {arr.shape}, dtype {arr.dtype}")
    print(f"  wrote {out}   <- open this in VS Code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
