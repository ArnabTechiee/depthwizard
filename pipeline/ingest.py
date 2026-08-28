"""
DepthWizard - Stage 1: Ingest.

Reads a PNG/JPG/GeoTIFF, extracts everything the metric pipeline needs
(CRS, ground sample distance in metres, sun azimuth/elevation), decides
whether the scene can be solved in ABSOLUTE metres or only RELATIVE units,
and writes a normalised working copy for the later stages.

Usage:
    python -m pipeline.ingest data/raw/scene.tif --max-size 1024
    python -m pipeline.ingest data/raw/photo.png            # relative mode
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning

import warnings
# Expected and handled: a plain PNG has no geotransform, that's the whole
# point of relative mode. Don't spam the console about it.
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


# --------------------------------------------------------------------------
# Metadata container
# --------------------------------------------------------------------------

@dataclass
class SceneMeta:
    """Everything stage 3 (calibration) needs to know about the input."""
    scene_id: str
    source_path: str
    mode: str                       # "absolute" | "relative"

    width: int
    height: int

    crs: Optional[str] = None       # e.g. "EPSG:32644"
    gsd_m: Optional[float] = None   # metres per pixel (mean of x/y)
    gsd_x_m: Optional[float] = None
    gsd_y_m: Optional[float] = None

    sun_elevation_deg: Optional[float] = None
    sun_azimuth_deg: Optional[float] = None

    center_lat: Optional[float] = None
    center_lon: Optional[float] = None

    acquisition_date: Optional[str] = None
    downsample_factor: float = 1.0
    metadata_source: Optional[str] = None   # where sun angles came from

    def blockers(self) -> list[str]:
        """Reasons this scene cannot produce a metric (absolute) DSM."""
        problems = []
        if self.gsd_m is None:
            problems.append("no GSD (image is not georeferenced)")
        if self.sun_elevation_deg is None:
            problems.append("no sun elevation (shadow calibration impossible)")
        if self.sun_azimuth_deg is None:
            problems.append("no sun azimuth (cannot orient shadow measurement)")
        return problems


# --------------------------------------------------------------------------
# Sun geometry extraction
# --------------------------------------------------------------------------

# Providers spell these differently. Match loosely, case-insensitively.
_ELEV_HINTS = ("sun_elevation", "sunel", "sun_el", "solar_elevation",
               "meansunel", "sun_angle_elevation", "sunelevation")
_AZIM_HINTS = ("sun_azimuth", "sunaz", "sun_az", "solar_azimuth",
               "meansunaz", "sun_angle_azimuth", "sunazimuth")
_DATE_HINTS = ("acquisition_date", "datetime", "date_acquired",
               "tifftag_datetime", "firstlinetime")


def _scan_tags(tags: dict, hints: tuple[str, ...]) -> Optional[str]:
    """Find the first tag whose key loosely matches one of the hints."""
    for key, value in tags.items():
        flat = key.lower().replace(" ", "").replace("-", "").replace("_", "")
        for hint in hints:
            if hint.replace("_", "") in flat:
                return value
    return None


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _read_sun_geometry(src) -> tuple[Optional[float], Optional[float],
                                     Optional[str], Optional[str]]:
    """Pull sun elevation/azimuth/date out of every tag namespace we can see."""
    merged: dict = {}
    merged.update(src.tags())
    for namespace in src.tag_namespaces():
        try:
            merged.update(src.tags(ns=namespace))
        except Exception:
            pass

    elevation = _as_float(_scan_tags(merged, _ELEV_HINTS))
    azimuth = _as_float(_scan_tags(merged, _AZIM_HINTS))
    date = _scan_tags(merged, _DATE_HINTS)

    source = "geotiff-tags" if (elevation is not None or azimuth is not None) else None
    return elevation, azimuth, date, source


# --------------------------------------------------------------------------
# Ground sample distance
# --------------------------------------------------------------------------

def _pixel_size_metres(src) -> tuple[Optional[float], Optional[float],
                                     Optional[float], Optional[float]]:
    """
    Return (gsd_x_m, gsd_y_m, center_lat, center_lon).

    A projected CRS gives metres directly. A geographic CRS (EPSG:4326)
    gives degrees, which must be converted at the scene's own latitude --
    a degree of longitude shrinks by cos(lat) and is 30% shorter in Delhi
    than at the equator. Getting this wrong scales every height you report.
    """
    if src.crs is None:
        return None, None, None, None

    transform = src.transform
    size_x, size_y = abs(transform.a), abs(transform.e)

    # Scene centre in native coordinates
    cx = transform.c + transform.a * (src.width / 2.0)
    cy = transform.f + transform.e * (src.height / 2.0)

    center_lat = center_lon = None
    try:
        from pyproj import Transformer
        to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        center_lon, center_lat = to_wgs84.transform(cx, cy)
    except Exception:
        pass

    if src.crs.is_geographic:
        lat = center_lat if center_lat is not None else 0.0
        metres_per_deg_lat = 111_132.0
        metres_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
        return (size_x * metres_per_deg_lon,
                size_y * metres_per_deg_lat,
                center_lat, center_lon)

    # Projected CRS - check the linear unit really is metres
    try:
        unit = src.crs.linear_units.lower()
        if "met" not in unit:
            print(f"  ! CRS linear unit is '{unit}', not metres. GSD is unreliable.",
                  file=sys.stderr)
    except Exception:
        pass

    return size_x, size_y, center_lat, center_lon


# --------------------------------------------------------------------------
# Sidecar overrides
# --------------------------------------------------------------------------

def _apply_sidecar(meta: SceneMeta, raw_path: Path) -> SceneMeta:
    """
    Look for <stem>.meta.json next to the image and let it override anything.

    Most freely-available imagery ships without sun angles in the tags. Rather
    than blocking the whole metric pipeline on that, supply them by hand:

        {"sun_elevation_deg": 52.3, "sun_azimuth_deg": 148.7, "gsd_m": 0.31}

    You can compute sun angles for any lat/lon/timestamp with pvlib or
    NOAA's solar calculator. Record where the number came from in the report.
    """
    sidecar = raw_path.with_suffix("")
    sidecar = sidecar.with_name(sidecar.name + ".meta.json")
    if not sidecar.exists():
        return meta

    with open(sidecar) as fh:
        override = json.load(fh)

    for field in ("sun_elevation_deg", "sun_azimuth_deg", "gsd_m",
                  "crs", "acquisition_date", "center_lat", "center_lon"):
        if override.get(field) is not None:
            setattr(meta, field, override[field])

    if override.get("gsd_m") is not None:
        # A hand-supplied GSD is assumed square and refers to the ORIGINAL
        # image, so re-apply the downsample factor we already used.
        g = float(override["gsd_m"]) * meta.downsample_factor
        meta.gsd_m = meta.gsd_x_m = meta.gsd_y_m = g

    meta.metadata_source = "sidecar" if meta.metadata_source is None else \
        f"{meta.metadata_source}+sidecar"
    print(f"  + applied overrides from {sidecar.name}")
    return meta


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def ingest(raw_path: Path, work_root: Path, max_size: int = 1024) -> SceneMeta:
    scene_id = raw_path.stem
    out_dir = work_root / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raw_path) as src:
        full_w, full_h = src.width, src.height

        # Decide the working resolution. Downsampling is not just for speed:
        # a 1000x1000 mesh renders fine in Three.js without any LOD work.
        factor = max(1.0, max(full_w, full_h) / float(max_size))
        out_w = int(round(full_w / factor))
        out_h = int(round(full_h / factor))

        bands = min(src.count, 3)
        rgb = src.read(
            indexes=list(range(1, bands + 1)),
            out_shape=(bands, out_h, out_w),
            resampling=Resampling.bilinear,
        )
        rgb = np.transpose(rgb, (1, 2, 0))          # HWC

        if rgb.dtype != np.uint8:
            # 11/16-bit satellite products: stretch on percentiles, not min/max,
            # so a few hot pixels don't crush the whole histogram.
            lo, hi = np.percentile(rgb.astype(np.float32), (2, 98))
            hi = hi if hi > lo else lo + 1.0
            rgb = np.clip((rgb.astype(np.float32) - lo) / (hi - lo), 0, 1)
            rgb = (rgb * 255).astype(np.uint8)

        if rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)

        gsd_x, gsd_y, lat, lon = _pixel_size_metres(src)
        elevation, azimuth, date, tag_source = _read_sun_geometry(src)

        # Resampling changes metres-per-pixel proportionally.
        if gsd_x is not None:
            gsd_x *= factor
            gsd_y *= factor

        meta = SceneMeta(
            scene_id=scene_id,
            source_path=str(raw_path),
            mode="relative",
            width=out_w,
            height=out_h,
            crs=str(src.crs) if src.crs else None,
            gsd_x_m=gsd_x,
            gsd_y_m=gsd_y,
            gsd_m=(gsd_x + gsd_y) / 2.0 if gsd_x is not None else None,
            sun_elevation_deg=elevation,
            sun_azimuth_deg=azimuth,
            center_lat=lat,
            center_lon=lon,
            acquisition_date=date,
            downsample_factor=factor,
            metadata_source=tag_source,
        )

    meta = _apply_sidecar(meta, raw_path)
    meta.mode = "relative" if meta.blockers() else "absolute"

    np.save(out_dir / "rgb.npy", rgb)
    with open(out_dir / "ingest.json", "w") as fh:
        json.dump(asdict(meta), fh, indent=2)

    return meta


def _report(meta: SceneMeta) -> None:
    print(f"\n  scene        : {meta.scene_id}")
    print(f"  working size : {meta.width} x {meta.height} "
          f"(downsampled {meta.downsample_factor:.2f}x)")
    print(f"  CRS          : {meta.crs or '-'}")
    if meta.gsd_m:
        print(f"  GSD          : {meta.gsd_m:.4f} m/px")
    else:
        print("  GSD          : -")
    if meta.center_lat is not None:
        print(f"  centre       : {meta.center_lat:.5f}, {meta.center_lon:.5f}")
    print(f"  sun elev/az  : "
          f"{meta.sun_elevation_deg if meta.sun_elevation_deg is not None else '-'} / "
          f"{meta.sun_azimuth_deg if meta.sun_azimuth_deg is not None else '-'}")
    print(f"  metadata src : {meta.metadata_source or 'none'}")
    print(f"\n  MODE         : {meta.mode.upper()}")

    if meta.mode == "absolute":
        # Sanity preview: how long a 10 m pole's shadow would be here.
        theta = math.radians(meta.sun_elevation_deg)
        px = 10.0 / math.tan(theta) / meta.gsd_m
        print(f"  sanity       : a 10 m structure casts a {px:.1f} px shadow")
    else:
        for reason in meta.blockers():
            print(f"  blocked by   : {reason}")
        print("  -> heights will be relative only. Add a .meta.json sidecar to fix.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="DepthWizard ingest stage")
    parser.add_argument("image", type=Path)
    parser.add_argument("--work", type=Path, default=Path("data/work"))
    parser.add_argument("--max-size", type=int, default=1024)
    args = parser.parse_args()

    if not args.image.exists():
        print(f"error: {args.image} not found", file=sys.stderr)
        return 1

    print(f"ingesting {args.image} ...")
    meta = ingest(args.image, args.work, args.max_size)
    _report(meta)
    print(f"wrote {args.work / meta.scene_id}/rgb.npy + ingest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
