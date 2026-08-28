"""
DepthWizard - fetch a real GeoTIFF from the Maxar Open Data Program.

Maxar publishes pre/post disaster imagery as public Cloud Optimized GeoTIFFs.
COG means we can read a *window* over HTTP instead of pulling a multi-GB tile,
so a usable crop costs ~30 MB and about a minute.

The script ranks candidate tiles on the three things that decide whether
shadow calibration will work at all:

  off-nadir   low is better. High off-nadir makes buildings lean in the
              image, displacing where the shadow appears to start.
  clouds      zero. Overcast means no shadows means no anchors.
  sun elev    30-55 deg is the sweet spot. Very high sun (Maxar tiles often
              sit at 70+) gives shadows so short they vanish into noise;
              very low sun gives long shadows that overlap everything.

Sun angles are written into the output GeoTIFF's tags, so `pipeline.ingest`
reports MODE: ABSOLUTE with no sidecar needed.

    python scripts/fetch_maxar.py --list
    python scripts/fetch_maxar.py --event Morocco-Earthquake-Sept-2023 --rank
    python scripts/fetch_maxar.py --event Morocco-Earthquake-Sept-2023 --pick 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

REPO_RAW = "https://raw.githubusercontent.com/opengeos/maxar-open-data/master"
API_DIR = "https://api.github.com/repos/opengeos/maxar-open-data/contents/datasets"

# Fallback if the GitHub API rate-limits you (it will, eventually).
KNOWN_EVENTS = [
    "Kahramanmaras-turkey-earthquake-23",
    "Morocco-Earthquake-Sept-2023",
    "Libya-Floods-Sept-2023",
    "Hurricane-Ian-9-26-2022",
    "Hurricane-Idalia-Aug-30-23",
    "Maui-Hawaii-fires-Aug-23",
    "Nepal-Earthquake-Nov-2023",
    "afghanistan-earthquake22",
    "BayofBengal-Cyclone-Mocha-May-23",
]

# Scoring preferences
SUN_ELEV_IDEAL = (30.0, 55.0)
MAX_OFF_NADIR = 25.0
MAX_CLOUD_PCT = 5.0



# Known urban centres worth aiming at, per event. Dense multi-storey cores
# give long unambiguous shadows; suburbs and rural tiles do not.
URBAN_HINTS = {
    "Kahramanmaras-turkey-earthquake-23": [
        ("Antakya", 36.2021, 36.1608),
        ("Kahramanmaras", 37.5858, 36.9371),
        ("Gaziantep", 37.0662, 37.3833),
    ],
    "Morocco-Earthquake-Sept-2023": [("Marrakech", 31.6295, -7.9811)],
    "Hurricane-Ian-9-26-2022": [
        ("Fort Myers", 26.6406, -81.8723),
        ("Cape Coral", 26.5629, -81.9495),
        ("Naples", 26.1420, -81.7948),
    ],
    "Libya-Floods-Sept-2023": [("Derna", 32.7627, 22.6367)],
}


def feature_centroid(feat: dict) -> tuple[float, float] | None:
    """
    Rough centroid of a footprint, in lon/lat.

    Handles both Polygon and MultiPolygon: the Maxar catalogs mix them, so
    walk the nested coordinate arrays down to the [lon, lat] pairs instead
    of assuming a fixed nesting depth.
    """
    geom = feat.get("geometry") or {}
    coords = geom.get("coordinates")
    if not coords:
        return None

    pts = []

    def walk(node):
        if (isinstance(node, (list, tuple)) and len(node) >= 2
                and all(isinstance(v, (int, float)) for v in node[:2])):
            pts.append((node[0], node[1]))
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(coords)
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts))


def km_between(lon1, lat1, lon2, lat2) -> float:
    import math
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * 111.320 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 110.574
    return math.hypot(dx, dy)


def list_events() -> list[str]:
    try:
        r = requests.get(API_DIR, timeout=30)
        data = r.json()
        if isinstance(data, list):
            names = sorted(
                item["name"][:-8] for item in data
                if item.get("name", "").endswith(".geojson")
            )
            if names:
                return names
    except Exception:
        pass
    print("  (GitHub API unavailable; showing built-in list)", file=sys.stderr)
    return KNOWN_EVENTS


def load_features(event: str) -> list[dict]:
    url = f"{REPO_RAW}/datasets/{event}.geojson"
    print(f"fetching catalog: {url}")
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        raise SystemExit(
            f"error: no catalog for '{event}' (HTTP {r.status_code}).\n"
            f"       run --list to see valid event names."
        )
    return r.json().get("features", [])


def score(props: dict) -> float | None:
    """Higher is better. None means reject outright."""
    off_nadir = props.get("view:off_nadir")
    clouds = props.get("tile:clouds_percent")
    sun_elev = props.get("view:sun_elevation")
    data_area = props.get("tile:data_area", 0) or 0

    if None in (off_nadir, sun_elev):
        return None

    # WV01 is panchromatic: one greyscale band. With no colour the
    # vegetation/water rejection in find_best_window is blind and will
    # happily call solid mangrove "built-up".
    if props.get("platform") not in ("WV02", "WV03", "GE01"):
        return None

    if off_nadir > MAX_OFF_NADIR:
        return None
    if clouds is not None and clouds > MAX_CLOUD_PCT:
        return None
    if data_area < 5.0:          # mostly-empty edge tile
        return None

    lo, hi = SUN_ELEV_IDEAL
    if lo <= sun_elev <= hi:
        sun_penalty = 0.0
    elif sun_elev < lo:
        sun_penalty = (lo - sun_elev) * 1.5
    else:
        # Overhead sun is the common failure here, penalise it harder.
        sun_penalty = (sun_elev - hi) * 2.0

    return 100.0 - sun_penalty - off_nadir * 1.2 - (clouds or 0) * 3.0


def rank(features: list[dict], limit: int = 12,
         near: tuple[float, float] | None = None,
         radius_km: float = 6.0) -> list[dict]:
    scored = []
    for feat in features:
        props = feat.get("properties", {})
        if not props.get("visual"):
            continue
        s = score(props)
        if s is None:
            continue

        dist = None
        if near is not None:
            centre = feature_centroid(feat)
            if centre is None:
                continue
            dist = km_between(centre[0], centre[1], near[1], near[0])
            if dist > radius_km:
                continue
            # Inside the radius, closeness to the city centre outweighs
            # a fraction of a degree of off-nadir.
            s += 40.0 * (1.0 - dist / radius_km)

        scored.append({"score": s, "props": props, "dist_km": dist})
    scored.sort(key=lambda d: -d["score"])
    return scored[:limit]


def show(candidates: list[dict], event: str) -> None:
    if not candidates:
        print("\nNo tile passed the filters.")
        hints = URBAN_HINTS.get(event)
        if hints:
            print("  Try widening --radius, or a different centre:")
            for name, lat, lon in hints:
                print(f"    --near {lat},{lon}    # {name}")
        else:
            print("  Try another event, or relax MAX_OFF_NADIR at the top of this file.")
        print()
        return

    print(f"\n  {'#':<3}{'sun_el':>8}{'off_nad':>9}{'cloud%':>8}"
          f"{'gsd':>7}{'plat':>7}{'km':>7}  date")
    print("  " + "-" * 69)
    for i, c in enumerate(candidates):
        p_ = c["props"]
        dist = c.get("dist_km")
        dist_s = f"{dist:.1f}" if dist is not None else "-"
        print(f"  {i:<3}{p_['view:sun_elevation']:>8.1f}"
              f"{p_['view:off_nadir']:>9.1f}"
              f"{p_.get('tile:clouds_percent', 0):>8.0f}"
              f"{p_.get('gsd', 0):>7.2f}"
              f"{p_.get('platform', '?'):>7}"
              f"{dist_s:>7}  {p_.get('datetime', '')[:10]}")
    print(f"\n  pick one with:  --pick <#>\n")


def find_best_window(src, size: int):
    """
    Locate the most useful `size` x `size` crop inside a sparse ARD tile.

    Maxar ARD tiles are grid squares, but the satellite strip only covers part
    of each one -- the rest is nodata black. Worse, the covered part is often
    empty terrain. Centring the crop is therefore wrong twice over.

    We read a cheap overview (COG overviews make this nearly free), score every
    candidate position on valid-pixel coverage plus local texture, and return
    the winner in full-resolution coordinates. Texture matters because a depth
    model has nothing to work with on blank desert, and shadow calibration
    needs buildings.
    """
    import numpy as np
    from rasterio.enums import Resampling

    max_ov = 1024
    factor = max(1.0, max(src.width, src.height) / max_ov)
    ov_w, ov_h = int(src.width / factor), int(src.height / factor)
    bands = min(src.count, 3)

    print(f"  scanning overview {ov_w} x {ov_h} for usable data ...")
    ov = src.read(
        indexes=list(range(1, bands + 1)),
        out_shape=(bands, ov_h, ov_w),
        resampling=Resampling.average,
    ).astype(np.float32)

    grey = ov.mean(axis=0)
    valid = (ov.max(axis=0) > 8)

    # Reject vegetation and water before scoring texture. Canopy mottling and
    # pond edges are high-variance but useless: neither gives a depth model a
    # height cue, and neither casts a measurable shadow. Without this the
    # picker happily lands on scrubland.
    if bands >= 3:
        r, g, b = ov[0], ov[1], ov[2]
        greenness = g - (r + b) / 2.0
        vegetation = greenness > 6.0
        water = (grey < 55.0) & (b >= r - 4.0)
    else:
        vegetation = np.zeros_like(grey, dtype=bool)
        water = grey < 55.0

    built = (valid & ~vegetation & ~water).astype(np.float64)
    veg_frac = float(vegetation[valid].mean()) if valid.any() else 0.0
    wat_frac = float(water[valid].mean()) if valid.any() else 0.0
    print(f"  tile composition: {veg_frac * 100:.0f}% vegetation, "
          f"{wat_frac * 100:.0f}% water")
    valid = valid.astype(np.float64)

    win = max(4, int(round(size / factor)))
    win = min(win, ov_h, ov_w)

    def box_sums(arr):
        """Sum over every win x win window, via an integral image."""
        integral = np.pad(arr, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return (integral[win:, win:] - integral[:-win, win:]
                - integral[win:, :-win] + integral[:-win, :-win])

    n = float(win * win)
    coverage = box_sums(valid) / n
    built_frac = box_sums(built) / n

    # Texture measured only over built-up pixels, so a park inside the window
    # cannot inflate the score.
    masked = grey * built
    mean = box_sums(masked) / np.maximum(box_sums(built), 1.0)
    sq = box_sums(masked * grey) / np.maximum(box_sums(built), 1.0)
    texture = np.sqrt(np.maximum(sq - mean ** 2, 0.0))
    texture = texture / (texture.max() or 1.0)

    # Built-up fraction dominates; texture breaks ties between built areas.
    score_map = built_frac + 0.3 * texture * (coverage > 0.85)

    idx = int(np.argmax(score_map))
    r_ov, c_ov = divmod(idx, score_map.shape[1])
    best_cov = float(coverage[r_ov, c_ov])
    best_built = float(built_frac[r_ov, c_ov])

    row_off = int(min(max(0, r_ov * factor), max(0, src.height - size)))
    col_off = int(min(max(0, c_ov * factor), max(0, src.width - size)))

    print(f"  best window: ({col_off}, {row_off})  coverage {best_cov * 100:.1f}%"
          f"  built-up {best_built * 100:.1f}%  texture {texture[r_ov, c_ov]:.2f}")
    if best_built < 0.35:
        print("  ! Little built-up surface here. Expect few shadow anchors.")
        print("    Use --near LAT,LON to aim at a city centre (--cities).")
    return row_off, col_off, best_cov


def download_window(props: dict, out_path: Path, size: int,
                    force: bool = False) -> None:
    """Read the best-covered window straight out of the remote COG."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    url = props["visual"]
    print(f"\nopening remote COG ...\n  {url}")

    # Remote range-request tuning. Without these GDAL can be slow or flaky
    # on large COGs over a home connection.
    gdal_env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_MAX_RETRY="5",
        GDAL_HTTP_RETRY_DELAY="2",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE="50000000",
    )

    try:
        with gdal_env, rasterio.open(f"/vsicurl/{url}") as src:
            print(f"  full tile: {src.width} x {src.height}, {src.count} bands, "
                  f"{src.dtypes[0]}")

            w = min(size, src.width)
            h = min(size, src.height)
            row_off, col_off, coverage = find_best_window(src, w)

            if coverage < 0.5 and not force:
                raise SystemExit(
                    f"\nAborting: best available window is only "
                    f"{coverage * 100:.1f}% real imagery.\n"
                    f"  This tile is mostly nodata. Try:\n"
                    f"    --pick 1   (or 2, 3 ...)   different tile\n"
                    f"    --size 1024                smaller crop fits better\n"
                    f"    --force                    download it anyway\n"
                )

            window = Window(col_off, row_off, w, h)
            print(f"  reading window {w} x {h} at ({col_off}, {row_off}) ...")
            bands = min(src.count, 3)
            data = src.read(indexes=list(range(1, bands + 1)), window=window)
            transform = src.window_transform(window)
            crs = src.crs
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            f"\nerror: could not read the remote COG.\n  {exc}\n\n"
            f"  Most likely your GDAL lacks curl support. Check with:\n"
            f"      python -c \"from osgeo import gdal; print(gdal.VersionInfo())\"\n"
            f"  Fix on Windows:  conda install -c conda-forge rasterio libgdal\n"
            f"  Or download the whole tile manually in a browser:\n      {url}\n"
        )

    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)

    actual = float((data.max(axis=0) > 8).mean())
    print(f"  verified coverage: {actual * 100:.1f}%")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=data.shape[1], width=data.shape[2], count=data.shape[0],
        dtype=data.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(data)
        # Bake sun geometry into the tags so ingest finds it automatically.
        dst.update_tags(
            SUN_ELEVATION=str(props["view:sun_elevation"]),
            SUN_AZIMUTH=str(props["view:sun_azimuth"]),
            ACQUISITION_DATE=props.get("datetime", ""),
            OFF_NADIR=str(props.get("view:off_nadir", "")),
            PLATFORM=props.get("platform", ""),
            SOURCE="Maxar Open Data Program (CC BY-NC 4.0)",
        )

    sidecar = out_path.with_suffix(".provenance.json")
    with open(sidecar, "w") as fh:
        json.dump(props, fh, indent=2)

    print(f"\nwrote {out_path}")
    print(f"      {sidecar}")
    print(f"\nnext:  python -m pipeline.ingest {out_path} --max-size 1024")
    print("       expect MODE: ABSOLUTE\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list disaster events")
    ap.add_argument("--event", type=str)
    ap.add_argument("--rank", action="store_true", help="show best tiles only")
    ap.add_argument("--pick", type=int, help="download candidate by index")
    ap.add_argument("--size", type=int, default=2048, help="crop size in px")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true",
                    help="download even if mostly nodata")
    ap.add_argument("--near", type=str, default=None,
                    help="target LAT,LON - e.g. --near 36.2021,36.1608")
    ap.add_argument("--radius", type=float, default=6.0,
                    help="km around --near (default 6)")
    ap.add_argument("--cities", action="store_true",
                    help="show known urban centres for this event")
    args = ap.parse_args()

    if args.list or not args.event:
        print("\navailable events:\n")
        for name in list_events():
            print("  " + name)
        print("\nthen:  --event <name> --rank\n")
        return 0

    if args.cities:
        hints = URBAN_HINTS.get(args.event)
        if not hints:
            print(f"\nno built-in centres for '{args.event}'.")
            print("find one on a map and pass --near LAT,LON\n")
            return 0
        print(f"\nurban centres in {args.event}:\n")
        for name, lat, lon in hints:
            print(f"  {name:<16} --near {lat},{lon}")
        print()
        return 0

    near = None
    if args.near:
        try:
            lat_s, lon_s = args.near.split(",")
            near = (float(lat_s), float(lon_s))
        except ValueError:
            print("error: --near must be LAT,LON", file=sys.stderr)
            return 1

    features = load_features(args.event)
    print(f"  {len(features)} tiles in catalog")
    candidates = rank(features, near=near, radius_km=args.radius)

    if args.pick is None:
        show(candidates, args.event)
        return 0

    if not 0 <= args.pick < len(candidates):
        print(f"error: --pick must be 0..{len(candidates) - 1}", file=sys.stderr)
        return 1

    props = candidates[args.pick]["props"]
    out = args.out or Path("data/raw") / f"{args.event}_{props['quadkey']}.tif"
    download_window(props, out, args.size, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
