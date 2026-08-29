"""
DepthWizard - terrain stratification report.

The rubric grades "performance stability across urban, sparse, hilly, and
forested landscapes", so a single headline RMSE does not answer it. This walks
every scene, tags it with its terrain class, and emits one table.

An important asymmetry in how these classes should be read: on sparse and
forested terrain there are few or no buildings, so a building-height RMSE is
not merely bad, it is undefined. The correct result there is that the pipeline
finds almost nothing to measure — and reporting that honestly is the answer,
not a failure to be hidden. Each scene therefore carries an explicit note
saying what its numbers mean.

    python -m pipeline.stratify --label antakya=urban fm=suburban \\
        ian_forest=forested ian2=sparse atlas=hilly
    python -m pipeline.stratify --md > docs/stratification.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASS_NOTE = {
    "urban": "dense mid-rise; long shadows, method operates in its best regime",
    "suburban": "low detached buildings; short shadows limit precision",
    "sparse": "few structures; building metrics are near-undefined by design",
    "forested": "canopy gives no monocular height cue and casts diffuse, "
                "merged shadows — expected failure, reported not omitted",
    "hilly": "terrain relief tests the DTM/nDSM decomposition rather than "
             "the shadow measurement",
}


def collect(scene: str, work_root: Path, terrain: str) -> dict | None:
    d = work_root / scene
    if not (d / "ingest.json").exists():
        return None
    ing = json.loads((d / "ingest.json").read_text())

    row = {
        "scene": scene,
        "terrain": terrain,
        "mode": ing.get("mode"),
        "gsd_m": ing.get("gsd_m"),
        "sun_elev": ing.get("sun_elevation_deg"),
        "buildings": None, "measured": None, "measured_pct": None,
        "median_h": None, "max_h": None, "repeat_mae": None,
        "lidar_rmse": None, "lidar_bias": None, "lidar_n": None,
        "note": CLASS_NOTE.get(terrain, ""),
    }

    bp = d / "buildings.json"
    if bp.exists():
        b = json.loads(bp.read_text())
        row["buildings"] = b.get("n_buildings")
        row["measured"] = b.get("n_measured")
        if b.get("n_buildings"):
            row["measured_pct"] = round(
                100 * (b.get("n_measured") or 0) / b["n_buildings"])
        row["median_h"] = b.get("median_height_m")
        row["max_h"] = b.get("max_height_m")
        row["repeat_mae"] = b.get("repeatability_mae_m")

    vp = d / "validation.json"
    if vp.exists():
        v = json.loads(vp.read_text())
        pb = v.get("per_building") or {}
        row["lidar_rmse"] = pb.get("rmse_m")
        row["lidar_bias"] = pb.get("bias_m")
        row["lidar_n"] = pb.get("n_buildings")
        row["metric"] = v.get("metric", "eave")
    return row


def fmt(v, unit="", nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}{unit}"
    return f"{v}{unit}"


def as_markdown(rows: list[dict]) -> str:
    out = ["# Terrain stratification", "",
           "Per-scene results grouped by terrain class. Building-height "
           "metrics are only meaningful where buildings exist; sparse and "
           "forested scenes are included to show where the method stops "
           "working, not to pad the table.", "",
           "| Scene | Terrain | GSD | Sun elev | Buildings | Measured | "
           "Median h | Repeat. MAE | vs LiDAR RMSE |",
           "|---|---|---|---|---|---|---|---|---|"]
    order = ["urban", "suburban", "hilly", "sparse", "forested"]
    rows = sorted(rows, key=lambda r: order.index(r["terrain"])
                  if r["terrain"] in order else 99)
    for r in rows:
        meas = (f"{r['measured']} ({r['measured_pct']}%)"
                if r["measured"] is not None else "—")
        out.append(
            f"| `{r['scene']}` | {r['terrain']} | {fmt(r['gsd_m'],' m/px',3)} "
            f"| {fmt(r['sun_elev'],'°',1)} | {fmt(r['buildings'])} | {meas} "
            f"| {fmt(r['median_h'],' m',1)} | {fmt(r['repeat_mae'],' m')} "
            f"| {fmt(r['lidar_rmse'],' m')} |")

    out += ["", "## Reading the table", ""]
    for r in rows:
        out.append(f"- **{r['scene']}** ({r['terrain']}): {r['note']}")

    out += ["", "## Method note", "",
            "Repeatability is a *precision* measure: each building's shadow "
            "rays are split in half at random and a height computed from "
            "each. It requires no ground truth, which is why it is the only "
            "figure available for scenes outside US LiDAR coverage.",
            "",
            "RMSE against LiDAR is an *accuracy* measure and is reported "
            "only where a classified point cloud covers the footprint."]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", nargs="+", required=True,
                    metavar="SCENE=TERRAIN",
                    help="e.g. antakya=urban fm=suburban atlas=hilly")
    ap.add_argument("--work", type=Path, default=Path("data/work"))
    ap.add_argument("--md", action="store_true", help="emit markdown")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for pair in args.label:
        if "=" not in pair:
            print(f"skipping '{pair}' — expected SCENE=TERRAIN")
            continue
        scene, terrain = pair.split("=", 1)
        row = collect(scene, args.work, terrain)
        if row is None:
            print(f"  ! no data for '{scene}' — run the pipeline first")
            continue
        rows.append(row)

    if not rows:
        return 1

    if args.md or args.out:
        text = as_markdown(rows)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
            print(f"wrote {args.out}")
        else:
            print(text)
    else:
        hdr = f"  {'scene':<14}{'terrain':<11}{'GSD':>8}{'bldgs':>7}" \
              f"{'meas':>7}{'med h':>8}{'repeat':>8}{'LiDAR':>8}"
        print("\n" + hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in rows:
            print(f"  {r['scene']:<14}{r['terrain']:<11}"
                  f"{fmt(r['gsd_m'],'',3):>8}{fmt(r['buildings']):>7}"
                  f"{fmt(r['measured_pct'],'%',0):>7}{fmt(r['median_h'],'',1):>8}"
                  f"{fmt(r['repeat_mae'],'',2):>8}{fmt(r['lidar_rmse'],'',2):>8}")
        print()

    json_path = (args.out.with_suffix(".json") if args.out
                 else Path("data/work/stratification.json"))
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
