"""
DepthWizard - Stage 2 (local): depth inference without Colab.

Runs Depth Anything V2 Small on the CPU. Slower than a T4, but it removes the
cloud round-trip and it is what makes offline deployment possible at all: the
PS asks for a standalone module, and a pipeline that phones home to Colab
is not standalone.

We use the transformers port rather than cloning the upstream repo, so the
whole thing is two pip packages and no vendored model code.

    python -m pipeline.depth_local --download          # fetch weights once
    python -m pipeline.depth_local antakya
    python -m pipeline.depth_local antakya fm --size 518

For Docker: run --download at BUILD time with HF_HOME pointing inside the
image, then set HF_HUB_OFFLINE=1 at runtime. Nothing touches the network.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

# Small is 25M params (~99 MB). Base and Large are better but the CPU cost
# rises steeply, and Small is enough: the depth map only supplies the ground
# surface and the footprint mask — the heights come from shadow geometry.
MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def load_model(size: str, weights_dir: Path | None):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError:
        raise SystemExit(
            "error: torch and transformers are required.\n\n"
            "  pip install --index-url https://download.pytorch.org/whl/cpu torch\n"
            "  pip install transformers\n\n"
            "  The CPU-only torch wheel is ~200 MB instead of ~2.5 GB."
        )

    if weights_dir:
        os.environ["HF_HOME"] = str(weights_dir.resolve())

    model_id = MODEL_IDS[size]
    print(f"  loading {model_id} ...")
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()

    torch.set_num_threads(max(1, (os.cpu_count() or 4)))
    print(f"  loaded in {time.time() - t0:.1f}s on CPU "
          f"({torch.get_num_threads()} threads)")
    return processor, model, torch


def infer(rgb: np.ndarray, processor, model, torch, input_size: int) -> np.ndarray:
    """Return a depth map at the input image's resolution."""
    from PIL import Image

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)

    image = Image.fromarray(rgb[:, :, :3])
    inputs = processor(images=image, size={"height": input_size,
                                           "width": input_size},
                       return_tensors="pt")

    with torch.no_grad():
        out = model(**inputs)

    # The model works at its own resolution; resample back to the scene grid
    # so every downstream stage keeps a 1:1 pixel correspondence with the RGB.
    depth = torch.nn.functional.interpolate(
        out.predicted_depth.unsqueeze(1),
        size=(rgb.shape[0], rgb.shape[1]),
        mode="bicubic", align_corners=False,
    ).squeeze()

    return depth.cpu().numpy().astype(np.float32)


def run(scene: str, work_root: Path, processor, model, torch,
        input_size: int) -> None:
    work_dir = work_root / scene
    rgb_path = work_dir / "rgb.npy"
    if not rgb_path.exists():
        print(f"  ! {rgb_path} not found — run pipeline.ingest first")
        return

    rgb = np.load(rgb_path)
    print(f"\n  {scene}: {rgb.shape[1]} x {rgb.shape[0]} ...", end="", flush=True)

    t0 = time.time()
    depth = infer(rgb, processor, model, torch, input_size)
    dt = time.time() - t0

    np.save(work_dir / "depth.npy", depth)
    print(f" done in {dt:.1f}s   range [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"  wrote {work_dir / 'depth.npy'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--work", type=Path, default=Path("data/work"))
    ap.add_argument("--model", choices=list(MODEL_IDS), default="small")
    ap.add_argument("--size", type=int, default=518,
                    help="model input resolution (518 native; 1036 sharper, 4x slower)")
    ap.add_argument("--weights-dir", type=Path, default=None,
                    help="cache weights here instead of the default HF cache")
    ap.add_argument("--download", action="store_true",
                    help="fetch weights and exit (use at Docker build time)")
    args = ap.parse_args()

    processor, model, torch = load_model(args.model, args.weights_dir)

    if args.download:
        cache = args.weights_dir or Path(os.environ.get(
            "HF_HOME", Path.home() / ".cache" / "huggingface"))
        print(f"\n  weights cached under {cache}")
        print("  set HF_HUB_OFFLINE=1 to run without network access\n")
        return 0

    if not args.scenes:
        print("\n  no scenes given. usage:")
        print("    python -m pipeline.depth_local antakya fm\n")
        return 1

    for scene in args.scenes:
        run(scene, args.work, processor, model, torch, args.size)

    print("\n  next: python -m pipeline.depth_check <scene> "
          "--orientation-from synthetic\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
