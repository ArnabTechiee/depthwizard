# Terrain stratification

Per-scene results grouped by terrain class. Building-height metrics are only meaningful where buildings exist; sparse and forested scenes are included to show where the method stops working, not to pad the table.

| Scene | Terrain | GSD | Sun elev | Buildings | Measured | Median h | Repeat. MAE | vs LiDAR RMSE |
|---|---|---|---|---|---|---|---|---|
| `antakya` | urban | 0.610 m/px | 28.4° | 167 | 125 (75%) | 9.0 m | 3.02 m | — |
| `fm` | suburban | 0.305 m/px | 37.3° | 150 | 106 (71%) | 2.4 m | 0.75 m | 3.66 m |
| `atlas` | hilly | 0.610 m/px | 33.5° | 50 | 18 (36%) | 6.5 m | 1.54 m | — |
| `ian2` | sparse | 0.610 m/px | 37.3° | 110 | 1 (1%) | 3.6 m | 3.20 m | — |
| `ian_forest` | forested | 0.610 m/px | 37.3° | 304 | 48 (16%) | 6.2 m | 2.30 m | — |

## Reading the table

- **antakya** (urban): dense mid-rise; long shadows, method operates in its best regime
- **fm** (suburban): low detached buildings; short shadows limit precision
- **atlas** (hilly): terrain relief tests the DTM/nDSM decomposition rather than the shadow measurement
- **ian2** (sparse): few structures; building metrics are near-undefined by design
- **ian_forest** (forested): canopy gives no monocular height cue and casts diffuse, merged shadows — expected failure, reported not omitted

## Method note

Repeatability is a *precision* measure: each building's shadow rays are split in half at random and a height computed from each. It requires no ground truth, which is why it is the only figure available for scenes outside US LiDAR coverage.

RMSE against LiDAR is an *accuracy* measure and is reported only where a classified point cloud covers the footprint.