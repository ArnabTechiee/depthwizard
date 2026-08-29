"""
DepthWizard - Stage 5: export to standard formats.

The PS asks for "a high-fidelity DSM in a standard geospatial format". Our
working arrays are .npy, which no GIS tool will open, so this writes the
deliverables a remote-sensing audience actually expects:

  <scene>_ndsm_eave.tif    height above ground, metres, float32, CRS-tagged
  <scene>_ndsm_ridge.tif   same with the eave->ridge roof correction applied
  <scene>_ndsm_ref.tif     LiDAR reference nDSM, when validation has been run
  <scene>_heatmap.png      quick-look colour render
  <scene>_buildings.csv    per-building table: id, centroid, area, heights
  <scene>_mesh.obj         textured 3D mesh (roofs + walls), with .mtl
  <scene>_metrics.json     accuracy figures in one place

A note on what these rasters contain: they are nDSM, i.e. height ABOVE LOCAL
GROUND, not absolute elevation above a vertical datum. That is deliberate --
it is the quantity we actually measure, it is datum-independent, and it is
what building-height analysis needs. Absolute elevation would require adding
a DTM and declaring a vertical datum; the field is set in the metadata so a
downstream user is never left guessing.

    python -m pipeline.export antakya
    python -m pipeline.export antakya fm --outdir exports
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


# --------------------------------------------------------------------------
# Raster
# --------------------------------------------------------------------------

def scene_transform(meta: dict) -> tuple[Affine, str]:
    """Geotransform for the working grid, scaled by ingest's downsampling."""
    with rasterio.open(meta["source_path"]) as src:
        base = src.transform
        crs = src.crs
    f = meta.get("downsample_factor", 1.0)
    return Affine(base.a * f, base.b, base.c,
                  base.d, base.e * f, base.f), crs


def write_geotiff(path: Path, array: np.ndarray, transform, crs,
                  description: str, extra_tags: dict | None = None) -> None:
    array = np.nan_to_num(array.astype(np.float32), nan=-9999.0)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=array.shape[0], width=array.shape[1], count=1,
        dtype="float32", crs=crs, transform=transform,
        nodata=-9999.0, compress="deflate", tiled=True,
    ) as dst:
        dst.write(array, 1)
        dst.set_band_description(1, description)
        dst.update_tags(
            SOFTWARE="DepthWizard (SIH26175)",
            METHOD="shadow-geometry height estimation from single-view optical",
            SURFACE_TYPE="nDSM (height above local ground, not absolute elevation)",
            UNITS="metres",
            **(extra_tags or {}),
        )
    print(f"    {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")


def write_heatmap(path: Path, array: np.ndarray) -> None:
    """Quick-look PNG with a perceptually ordered ramp."""
    from PIL import Image

    finite = np.isfinite(array) & (array > -1000)
    if not finite.any():
        return
    lo, hi = 0.0, float(np.percentile(array[finite], 99.5)) or 1.0
    t = np.clip((array - lo) / (hi - lo), 0, 1)

    # Dark blue -> cyan -> yellow -> white. Monotonic in luminance so it also
    # reads correctly when printed in greyscale.
    stops = np.array([[13, 20, 60], [20, 110, 160], [70, 190, 160],
                      [230, 210, 90], [255, 255, 255]], dtype=np.float32)
    idx = t * (len(stops) - 1)
    lo_i = np.floor(idx).astype(int).clip(0, len(stops) - 2)
    frac = (idx - lo_i)[..., None]
    rgb = stops[lo_i] * (1 - frac) + stops[lo_i + 1] * frac

    Image.fromarray(rgb.astype(np.uint8)).save(path)
    print(f"    {path.name}")


# --------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------

def write_obj(obj_path: Path, heights: np.ndarray, span_m: float,
              texture_name: str, wall_threshold: float, stride: int) -> None:
    """
    Blocky mesh with real vertical walls, matching what the viewer renders.

    A smooth height-field has no vertical faces, so building sides come out as
    stretched roof texture. Emitting one flat quad per cell plus a wall quad at
    every height discontinuity gives geometry that survives a ground-level
    camera in Blender, QGIS or any glTF pipeline.
    """
    h = heights[::stride, ::stride]
    n_rows, n_cols = h.shape
    cell_x = span_m / n_cols
    cell_z = span_m / n_rows
    half_x, half_z = span_m / 2.0, span_m / 2.0

    verts: list[str] = []
    uvs: list[str] = []
    faces: list[str] = []
    walls = 0

    def add_quad(p, t=None):
        base = len(verts) + 1
        for (x, y, z) in p:
            verts.append(f"v {x:.3f} {y:.3f} {z:.3f}")
        if t:
            tb = len(uvs) + 1
            for (u, v) in t:
                uvs.append(f"vt {u:.5f} {v:.5f}")
            faces.append(f"f {base}/{tb} {base+1}/{tb+1} {base+2}/{tb+2}")
            faces.append(f"f {base}/{tb} {base+2}/{tb+2} {base+3}/{tb+3}")
        else:
            faces.append(f"f {base} {base+1} {base+2}")
            faces.append(f"f {base} {base+2} {base+3}")

    for r in range(n_rows):
        for c in range(n_cols):
            y = float(h[r, c])
            x0 = -half_x + c * cell_x
            x1 = x0 + cell_x
            z0 = -half_z + r * cell_z
            z1 = z0 + cell_z
            u0, u1 = c / n_cols, (c + 1) / n_cols
            v0, v1 = 1 - r / n_rows, 1 - (r + 1) / n_rows

            add_quad([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)],
                     [(u0, v0), (u1, v0), (u1, v1), (u0, v1)])

            if c + 1 < n_cols:
                hr = float(h[r, c + 1])
                if abs(y - hr) > wall_threshold:
                    lo, hi = min(y, hr), max(y, hr)
                    add_quad([(x1, lo, z0), (x1, lo, z1), (x1, hi, z1), (x1, hi, z0)])
                    walls += 1
            if r + 1 < n_rows:
                hd = float(h[r + 1, c])
                if abs(y - hd) > wall_threshold:
                    lo, hi = min(y, hd), max(y, hd)
                    add_quad([(x0, lo, z1), (x1, lo, z1), (x1, hi, z1), (x0, hi, z1)])
                    walls += 1

    mtl_path = obj_path.with_suffix(".mtl")
    mtl_path.write_text(
        "newmtl depthwizard\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\n"
        f"map_Kd {texture_name}\n"
    )

    with open(obj_path, "w") as fh:
        fh.write("# DepthWizard mesh — Y is height in metres, X/Z are ground "
                 "metres from the scene centre\n")
        fh.write(f"mtllib {mtl_path.name}\nusemtl depthwizard\n")
        fh.write("\n".join(verts) + "\n")
        fh.write("\n".join(uvs) + "\n")
        fh.write("\n".join(faces) + "\n")

    tris = len(faces)
    print(f"    {obj_path.name}  ({obj_path.stat().st_size / 1e6:.1f} MB, "
          f"{tris:,} tris, {walls} walls)")
    print(f"    {mtl_path.name}")


# --------------------------------------------------------------------------

def run(scene: str, work_root: Path, out_root: Path, stride: int) -> None:
    work_dir = work_root / scene
    out_dir = out_root / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((work_dir / "ingest.json").read_text())
    transform, crs = scene_transform(meta)

    print(f"\n  {scene}  ({meta['width']} x {meta['height']}, "
          f"{meta['gsd_m']:.4f} m/px, {crs})")

    eave = np.load(work_dir / "ndsm_m.npy").astype(np.float32)
    tags = {"SUN_ELEVATION_DEG": str(meta.get("sun_elevation_deg")),
            "SUN_AZIMUTH_DEG": str(meta.get("sun_azimuth_deg")),
            "ACQUISITION_DATE": str(meta.get("acquisition_date") or "")}

    write_geotiff(out_dir / f"{scene}_ndsm_eave.tif", eave, transform, crs,
                  "nDSM eave height (m)", {**tags, "HEIGHT_REFERENCE": "eave"})

    ridge_path = work_dir / "ndsm_ridge_m.npy"
    if ridge_path.exists():
        write_geotiff(out_dir / f"{scene}_ndsm_ridge.tif",
                      np.load(ridge_path), transform, crs,
                      "nDSM ridge height (m)",
                      {**tags, "HEIGHT_REFERENCE": "ridge (roof-pitch prior)"})

    ref_path = work_dir / "ndsm_ref.npy"
    if ref_path.exists():
        write_geotiff(out_dir / f"{scene}_ndsm_ref.tif",
                      np.load(ref_path), transform, crs,
                      "LiDAR reference nDSM (m)",
                      {"SOURCE": "USGS 3DEP LiDAR point cloud"})

    write_heatmap(out_dir / f"{scene}_heatmap.png", eave)

    # Per-building CSV — the structured product a GIS user can join on.
    bpath = work_dir / "buildings.json"
    if bpath.exists():
        data = json.loads(bpath.read_text())
        rows = data.get("buildings", [])
        csv_path = out_dir / f"{scene}_buildings.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "easting_m", "northing_m", "area_m2",
                        "eave_height_m", "ridge_height_m", "roof_rise_m",
                        "floors_est", "method", "n_rays", "repeatability_m"])
            for b in rows:
                east, north = transform * (b["col"], b["row"])
                h = b.get("height_m")
                w.writerow([
                    b["id"], f"{east:.2f}", f"{north:.2f}", b.get("area_m2"),
                    h, b.get("ridge_m"), b.get("roof_rise_m"),
                    round(h / 3.0, 1) if h else None,
                    b.get("method"), b.get("n_rays"), b.get("repeatability_m"),
                ])
        print(f"    {csv_path.name}  ({len(rows)} buildings)")

    # Mesh, sharing the viewer's texture
    tex_src = out_root.parent / "viewer" / "data" / scene / "texture.jpg"
    tex_name = f"{scene}_texture.jpg"
    if tex_src.exists():
        (out_dir / tex_name).write_bytes(tex_src.read_bytes())
    write_obj(out_dir / f"{scene}_mesh.obj", eave,
              meta["gsd_m"] * meta["width"], tex_name,
              wall_threshold=1.0, stride=stride)

    metrics = {"scene": scene, "crs": str(crs), "gsd_m": meta["gsd_m"],
               "extent_m": round(meta["gsd_m"] * meta["width"], 1),
               "sun_elevation_deg": meta.get("sun_elevation_deg"),
               "sun_azimuth_deg": meta.get("sun_azimuth_deg")}
    for name in ("buildings.json", "validation.json", "ground.json"):
        p = work_dir / name
        if p.exists():
            d = json.loads(p.read_text())
            metrics[name.replace(".json", "")] = {
                k: v for k, v in d.items() if k != "buildings"}
    (out_dir / f"{scene}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"    {scene}_metrics.json")
    print(f"  -> {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--work", type=Path, default=Path("data/work"))
    ap.add_argument("--outdir", type=Path, default=Path("exports"))
    ap.add_argument("--mesh-stride", type=int, default=2,
                    help="subsample factor for the OBJ (2 keeps files sane)")
    args = ap.parse_args()

    for scene in args.scenes:
        run(scene, args.work, args.outdir, args.mesh_stride)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
