"""
DepthWizard - the application layer.

The PS asks for an "Interactive Visualization Platform" that lets users
"upload imagery, visualize reconstructed terrain, and validate estimated
height values against reference datasets". A CLI plus a static viewer is not
that. This wraps the pipeline in an HTTP service so a GeoTIFF dropped in the
browser comes back as a navigable 3D scene.

Both input modes required by the PS are handled:

  GeoTIFF with CRS + sun geometry -> ABSOLUTE DSM, heights in metres
  PNG/JPG with no spatial metadata -> RELATIVE DSM (rDSM), unitless

The depth model is loaded once at startup, not per request: it is ~390 MB and
reloading it per upload would dominate the response time.

    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import shutil
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
WORK = ROOT / "data" / "work"
VIEWER = ROOT / "viewer"
ALLOWED = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

app = FastAPI(title="DepthWizard", version="1.0")

JOBS: dict[str, dict] = {}
_model_lock = threading.Lock()
_model = None


def get_model(size: str = "base"):
    """Load the depth backbone once and share it across requests."""
    global _model
    with _model_lock:
        if _model is None or _model[0] != size:
            from pipeline.depth_local import load_model
            processor, net, torch = load_model(size, None)
            _model = (size, processor, net, torch)
    return _model[1], _model[2], _model[3]


# --------------------------------------------------------------------------
# Pipeline job
# --------------------------------------------------------------------------

STAGES = ["ingest", "depth", "orient", "ground", "calibrate", "bake"]


def _set(job_id: str, **kw):
    JOBS[job_id].update(kw)


def run_pipeline(job_id: str, scene: str, image_path: Path,
                 max_size: int, model_size: str, min_area: float) -> None:
    import numpy as np
    from pipeline import ingest as m_ingest
    from pipeline import depth_local as m_depth
    from pipeline import depth_check as m_check
    from pipeline import ground as m_ground
    from pipeline import planb as m_planb
    from pipeline import bake as m_bake

    try:
        _set(job_id, stage="ingest", progress=1 / len(STAGES))
        meta = m_ingest.ingest(image_path, WORK, max_size)
        _set(job_id, mode=meta.mode, gsd_m=meta.gsd_m,
             blockers=meta.blockers())

        _set(job_id, stage="depth", progress=2 / len(STAGES))
        processor, net, torch = get_model(model_size)
        rgb = np.load(WORK / scene / "rgb.npy")
        depth = m_depth.infer(rgb, processor, net, torch, 518)
        np.save(WORK / scene / "depth.npy", depth)

        _set(job_id, stage="orient", progress=3 / len(STAGES))
        ref = WORK / "synthetic" / "depth.json"
        orientation = None
        if ref.exists():
            import json
            orientation = json.loads(ref.read_text()).get("orientation")
        m_check.analyse(scene, WORK, None, orientation or 1)

        _set(job_id, stage="ground", progress=4 / len(STAGES))
        m_ground.run(scene, WORK, 60.0, None)

        _set(job_id, stage="calibrate", progress=5 / len(STAGES))
        if meta.mode == "absolute":
            result = m_planb.run(scene, WORK, min_area, 0.0)
            _set(job_id, buildings=result.get("n_buildings"),
                 measured=result.get("n_measured"),
                 median_height_m=result.get("median_height_m"),
                 repeatability_mae_m=result.get("repeatability_mae_m"))
        else:
            # rDSM path: no GSD or sun angles, so no metric anchor exists.
            # We still deliver a relative surface — the PS asks for exactly
            # this for non-georeferenced input — but we must not dress
            # unitless numbers up as metres.
            rel = np.load(WORK / scene / "ndsm_rel.npy")
            span = float(rel.max()) or 1.0
            np.save(WORK / scene / "ndsm_m.npy",
                    (rel / span).astype(np.float32) * 100.0)
            _set(job_id, note="relative DSM — values are unitless (0-100), "
                              "not metres")

        _set(job_id, stage="bake", progress=6 / len(STAGES))
        m_bake.run(scene, WORK, VIEWER / "data", 512)

        _set(job_id, stage="done", progress=1.0, status="done",
             finished=datetime.utcnow().isoformat(timespec="seconds"),
             viewer_url=f"/viewer/?scene={scene}")
    except Exception as exc:
        _set(job_id, status="error", stage="failed",
             error=f"{type(exc).__name__}: {exc}",
             traceback=traceback.format_exc()[-2000:])


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

@app.post("/api/scenes")
async def upload(file: UploadFile = File(...),
                 max_size: int = Form(1024),
                 model: str = Form("base"),
                 min_area: float = Form(120.0)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"unsupported type '{suffix}'. "
                                 f"accepted: {', '.join(sorted(ALLOWED))}")

    scene = Path(file.filename).stem.replace(" ", "_")[:40] or "scene"
    # Never clobber an existing scene: an upload that silently destroys a
    # validated result is a failure mode you only notice afterwards.
    if (WORK / scene).exists():
        scene = f"{scene}_{uuid.uuid4().hex[:4]}"

    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / f"{scene}{suffix}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"id": job_id, "scene": scene, "status": "running",
                    "stage": "queued", "progress": 0.0,
                    "started": datetime.utcnow().isoformat(timespec="seconds")}

    threading.Thread(target=run_pipeline, daemon=True,
                     args=(job_id, scene, dest, max_size, model, min_area)
                     ).start()
    return {"job_id": job_id, "scene": scene, "stages": STAGES}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[job_id]


@app.get("/api/scenes")
async def list_scenes():
    import json
    out = []
    base = VIEWER / "data"
    if base.exists():
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            meta = d / "meta.json"
            if not meta.exists():
                continue
            m = json.loads(meta.read_text())
            out.append({"scene": m.get("scene", d.name),
                        "buildings": len(m.get("buildings", [])),
                        "max_height_m": m.get("max_height_m"),
                        "gsd_m": m.get("gsd_m"),
                        "stats": m.get("stats", {})})
    return out


@app.get("/api/scenes/{scene}/exports")
async def export_scene(scene: str):
    """Generate the standard-format deliverables on demand."""
    from pipeline import export as m_export
    if not (WORK / scene / "ndsm_m.npy").exists():
        raise HTTPException(404, f"scene '{scene}' has no DSM yet")
    m_export.run(scene, WORK, ROOT / "exports", stride=4)
    files = sorted(p.name for p in (ROOT / "exports" / scene).iterdir())
    return {"scene": scene, "files": files,
            "download": f"/exports/{scene}/"}


@app.get("/api/health")
async def health():
    return {"ok": True, "model_loaded": _model is not None,
            "jobs": len(JOBS)}


@app.get("/")
async def index():
    return FileResponse(VIEWER / "upload.html")


for name, path in (("viewer", VIEWER), ("exports", ROOT / "exports")):
    path.mkdir(parents=True, exist_ok=True)
    app.mount(f"/{name}", StaticFiles(directory=path, html=True), name=name)