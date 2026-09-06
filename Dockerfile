# DepthWizard - fully offline single-view height estimation.
#
# The PS asks for a standalone module. For an ISRO evaluation that means it
# must run with the network cable pulled: model weights baked in at build
# time, Three.js served locally, sample scenes vendored. Nothing is fetched
# at runtime.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/weights

WORKDIR /app

# rasterio's compiled extension dynamically links libexpat at runtime (an
# XML-parsing C library, pulled in transitively via GDAL). It is stripped
# out of python:3.12-slim to keep the base image small, so it must be
# installed explicitly or rasterio fails at import time with
# "ImportError: libexpat.so.1: cannot open shared object file" --
# even though `pip install rasterio` itself succeeds silently, because
# pip only installs the Python wheel, not this system library.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 \
 && rm -rf /var/lib/apt/lists/*

# rasterio, laspy and torch otherwise ship manylinux wheels with their
# native libraries bundled, so no further system GDAL/PROJ install is
# required beyond the one line above.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
 && pip install -r requirements.txt

COPY pipeline/ ./pipeline/
COPY scripts/  ./scripts/
COPY viewer/   ./viewer/
COPY app.py    ./app.py

# Pre-baked demo scenes so the viewer has something to show immediately,
# before anyone uploads a file. Only the baked outputs are needed for that --
# NOT the multi-hundred-MB source GeoTIFFs, which stay out of the image.
COPY viewer/data/ ./viewer/data/

# The synthetic regression scene's source + ground truth ARE small and are
# needed by the pipeline itself (orientation reference in depth_check,
# regression test), so those two files are copied by name rather than
# pulling in all of data/raw/.
COPY data/raw/synthetic.tif       ./data/raw/synthetic.tif
COPY data/raw/synthetic.truth.npy ./data/raw/synthetic.truth.npy
COPY data/work/synthetic/depth.json ./data/work/synthetic/depth.json

# Bake the model weights into the image. This is the step that makes the
# container airgap-safe -- it needs real network access to fetch weights
# ONCE at build time. HF_HUB_OFFLINE is deliberately NOT set yet: setting
# it before this line makes transformers refuse to look at the network at
# all, which fails immediately since the local cache is still empty.
RUN python -m pipeline.depth_local --download --model base \
 && python -c "import pathlib;p=pathlib.Path('/app/weights');\
print('weights:', round(sum(f.stat().st_size for f in p.rglob('*') if f.is_file())/1e6,1), 'MB')"

# From here on, HF_HUB_OFFLINE=1 -- this is what proves the container is
# airgap-safe. Anything that tries to reach the network after this point
# fails loudly instead of silently working here and breaking at the venue.
ENV HF_HUB_OFFLINE=1

# Writable areas the running app needs: new uploads, new pipeline output,
# generated exports. Declared as volumes so a container restart doesn't
# lose work in progress, and so docker-compose can mount them persistently.
RUN mkdir -p data/work exports
VOLUME ["/app/data/work", "/app/exports"]

EXPOSE 8000

# Serves the actual interactive platform -- upload, background pipeline
# jobs, live progress, the 3D viewer, and the export endpoint -- not just
# static files. This is what makes the "Interactive Visualization Platform"
# requirement literally true inside the container.
#
# Invoked as `python -m uvicorn` rather than the bare `uvicorn` command:
# the console-script entry point is not reliably on $PATH in every base
# image/pip configuration, but `python -m` always finds an installed
# package regardless of PATH.
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
