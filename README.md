# DepthWizard

**Building heights from a single satellite image, using shadow geometry.**

Smart India Hackathon 2026 · Problem Statement **SIH26175** · ISRO / Space
Applications Centre · Theme: Disaster Management

---

## What it does

Takes one flat optical satellite image — no LiDAR, no stereo pair, no second
viewing angle — and produces a metric Digital Surface Model plus a navigable
textured 3D scene.

Heights are **measured, not inferred**. For a vertical structure whose shadow
falls on flat ground:

```
h  =  L_pixels  ×  GSD  ÷  tan(sun_elevation)
```

Sun elevation and azimuth come from the product's own metadata; ground sample
distance comes from the GeoTIFF affine transform. Every isolated building in
the scene is therefore a free absolute anchor derived from physics rather than
a learned prior.

**Why it matters:** after a flood, landslide or earthquake you need to know
what the terrain and buildings look like *now*, in hours rather than weeks.
LiDAR flights and InSAR tasking are expensive, sensor-dependent and slow. A
single optical frame is cheap and already being captured almost anywhere.

---

## Results

| Scene | Buildings | Measured from shadow | Median height | Repeatability MAE |
|---|---|---|---|---|
| Antakya, Türkiye (dense urban) | 167 | 125 (75%) | 9.0 m | 3.02 m |
| Fort Myers, USA (suburban) | 150 | 106 (71%) | 2.4 m | 0.75 m |

### Validated against LiDAR

Fort Myers, compared per-building against USGS 3DEP LiDAR (n=150, 70% reference
coverage):

| Surface compared | RMSE | MAE | Bias | Correlation |
|---|---|---|---|---|
| Eave height | 4.12 m | 3.62 m | −2.89 m | 0.305 |
| Ridge height | **3.66 m** | **2.97 m** | **−0.22 m** | 0.183 |

Shadow geometry measures to the **eave** — the shadow is cast by the roof's
outer edge. LiDAR's highest return is the **ridge**. Both are correct; they
measure different surfaces. Applying a stated architectural prior (a 5:12
residential pitch over each building's own short span) reduces bias from
−2.89 m to −0.22 m. The prior is geometric and is *not* fitted to the
reference data.

---

## Approach, and one thing that did not work

The obvious route is a pretrained monocular depth model. We tried that first
with Depth Anything V2, then validated it against held-out shadow anchors on a
dense urban scene and measured **R² = −0.74** — worse than predicting the mean.
The depth value for a building carried no usable information about its height.

So the depth model was demoted. It still supplies the ground surface and the
building footprint mask, but the metres come from each building's own shadow.
That decision is recorded here because a measured choice is worth more than an
assumed one, and because the fallback was written down before it was needed.

### Model ablation

| Model | Device | fm RMSE | Measured | Antakya buildings |
|---|---|---|---|---|
| DAv2 Large | GPU (Colab) | 3.30 m | 116/153 | 201 |
| DAv2 Base | CPU, offline | 4.21 m | 92/146 | 167 |
| DAv2 Small | CPU, offline | 4.32 m | 23/95 | 115 |

Larger backbones resolve more structure, so more footprints survive
segmentation. Running fully offline on CPU costs roughly 0.9 m of RMSE against
GPU Large — a quantified trade against the PS's standalone-deployment
requirement rather than a hidden compromise.

---

## Pipeline

| Stage | Module | Output |
|---|---|---|
| Ingest | `pipeline.ingest` | CRS, GSD in metres, sun geometry, ABSOLUTE/RELATIVE mode |
| Depth | `pipeline.depth_local` | Relative inverse-depth map (CPU, offline) |
| Orientation check | `pipeline.depth_check` | Sign convention settled against known truth |
| Ground separation | `pipeline.ground` | DTM / nDSM split by morphological opening |
| Metric calibration | `pipeline.planb` | Per-building shadow heights, eave and ridge |
| Validation | `pipeline.validate` | RMSE / MAE / bias vs LiDAR point cloud |
| Bake | `pipeline.bake` | Static viewer assets |
| Viewer | `viewer/index.html` | Textured 3D scene, height probe, flood model |

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

Windows note: if `pip install rasterio` fails, use
`conda install -c conda-forge rasterio` or a wheel from
[cgohlke/geospatial-wheels](https://github.com/cgohlke/geospatial-wheels).

## For team members

**Just want to see it?** Baked viewer scenes are committed. No pipeline run, no
downloads:

```bash
python -m http.server 8000        # then open localhost:8000/viewer/
```

**Want to re-run the pipeline?** The source GeoTIFFs are committed too, so
everything downstream regenerates in about two minutes:

```bash
python -m pipeline.depth_local antakya fm --model base
python -m pipeline.depth_check antakya --orientation-from synthetic
python -m pipeline.depth_check fm      --orientation-from synthetic
python -m pipeline.ground      antakya --max-building-m 60
python -m pipeline.ground      fm      --max-building-m 60
python -m pipeline.planb       antakya --min-area 120
python -m pipeline.planb       fm      --min-area 120
python -m pipeline.bake        antakya fm --grid 512
```

Every stage reads the previous stage's output from disk, so **skipping one
silently reuses stale data** rather than failing. Run them in order.

**Want to reproduce the LiDAR validation?** The point clouds are not in the repo
(231 MB). Download these four tiles from
[The National Map](https://apps.nationalmap.gov/downloader) into `data/lidar/`
— *Elevation Source Data (3DEP) → Lidar Point Cloud (LPC)*:

```
USGS_LPC_FL_Southwest_2018_D18_SUPPLEMENTAL_w1428n4840.laz
USGS_LPC_FL_Southwest_2018_D18_SUPPLEMENTAL_w1428n4850.laz
USGS_LPC_FL_Southwest_2018_D18_SUPPLEMENTAL_w1429n4840.laz
USGS_LPC_FL_Southwest_2018_D18_SUPPLEMENTAL_w1429n4850.laz
```

The Fort Myers scene straddles all four, so pass them together:

```bash
python -m pipeline.validate fm --metric ridge --lidar data/lidar/*.laz
```

On PowerShell, wildcards are not expanded for external programs — use
`--lidar (Get-ChildItem data\lidar\*.laz).FullName` or list the four paths.

### Run the synthetic regression scene

```bash
python scripts/make_sample.py --out data/raw/synthetic.tif
python -m pipeline.ingest data/raw/synthetic.tif --max-size 1024
```

This generates eight buildings of known height with physically correct cast
shadows, and a truth file recording the exact shadow length each should
produce. It is the unit test for the shadow formula and the GSD conversion —
keep it in the repo permanently.

### Run a real scene end to end

```bash
python scripts/fetch_maxar.py --event Kahramanmaras-turkey-earthquake-23 --cities
python scripts/fetch_maxar.py --event Kahramanmaras-turkey-earthquake-23 \
       --near 36.2021,36.1608 --radius 8 --pick 0 --size 2048 --out data/raw/antakya.tif

python -m pipeline.ingest      data/raw/antakya.tif --max-size 1024
python -m pipeline.depth_local antakya --model base
python -m pipeline.depth_check antakya --orientation-from synthetic
python -m pipeline.ground      antakya --max-building-m 60
python -m pipeline.planb       antakya --min-area 120
python -m pipeline.bake        antakya --grid 512

python -m http.server 8000     # then open localhost:8000/viewer/?scene=antakya
```

### Validate against LiDAR

Download a classified point cloud for the scene footprint from
[The National Map](https://apps.nationalmap.gov/downloader) — *Elevation
Source Data (3DEP) → Lidar Point Cloud (LPC)*. Downloadable 3DEP **DEM**
products are bare-earth and cannot validate building heights; you need the
point cloud.

```bash
python -m pipeline.validate fm --metric ridge --lidar data/lidar/*.laz
```

Comparison is nDSM against nDSM — height above ground — so the
EGM96-vs-WGS84 vertical datum offset cancels out entirely rather than
injecting tens of metres.

---

## Offline deployment

```bash
python scripts/vendor_three.py       # serve Three.js locally, not from a CDN
docker compose build                 # bakes model weights into the image
docker compose up
```

Set `network_mode: none` in `docker-compose.yml` to verify it runs airgapped.
`HF_HUB_OFFLINE=1` is set inside the image, so any attempt to reach the network
fails loudly rather than silently succeeding on a developer machine.

---

## Known limits

- **Forested terrain fails.** Canopy provides almost no monocular height cue,
  and C-band reference data is itself contaminated by partial penetration.
  Reported honestly rather than omitted.
- **Correlation is weak where height variance is low.** Fort Myers buildings
  cluster at 3–8 m, leaving little spread to detect. Antakya spans 4–29 m and
  is where the method shows range.
- **Short shadows limit accuracy.** At 0.61 m/px a 3 m building casts a ~10 px
  shadow. Re-ingesting at native 0.305 m/px improved measured coverage from 63%
  to 71% and halved repeatability RMSE. The fix is finer imagery, not a better
  model.
- **Off-nadir imagery causes building lean**, displacing where each shadow
  appears to begin. Tiles are filtered to low off-nadir where possible.
- **Antakya has no ground truth.** That scene reports repeatability
  (a precision measure), not accuracy.

## Out of scope

Stated deliberately: multi-view or stereo reconstruction, SAR/InSAR fusion,
real-time video ingest, atmospheric correction, and any survey-grade accuracy
claim.

---

## Data sources

- [Maxar Open Data Program](https://github.com/opengeos/maxar-open-data) — pre/post disaster imagery (CC BY-NC 4.0)
- [USGS 3DEP](https://apps.nationalmap.gov/downloader) — LiDAR point clouds for validation
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — monocular depth backbone