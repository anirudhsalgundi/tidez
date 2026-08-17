"""
Web-based interactive scanner for the tidez master H5.

Serves an interactive lightcurve (Plotly) for each object; clicking a DECam
data point loads its template/science/diff cutouts below. Prev/Next step
through objects, an objid-file field restricts the browsing list, and
favoriting works the same as tidez/lightcurve_viewer.py.

Usage:
    python -m tidez.webapp
    python -m tidez.webapp --h5 data/tidez_master.h5 --port 8000
"""

import argparse
import pathlib

import math

import h5py
import uvicorn
from astropy.time import Time
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from tidez.cutouts import get_object_cutouts
from tidez.logger import get_logger
from tidez.plot_lightcurve import (
    DEFAULT_H5,
    crossmatch_title_suffix,
    first_detection_jd,
    load_object_rows,
)

logger = get_logger("webapp")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
DEFAULT_FAVORITES_FILE = DATA_DIR / "favorite_sources.txt"

app = FastAPI(title="tidez scanner")

# Set by main() before uvicorn.run(); read by the request handlers below.
STATE = {"h5_path": DEFAULT_H5, "favorites_file": DEFAULT_FAVORITES_FILE}


def _json_safe(value):
    return None if isinstance(value, float) and math.isnan(value) else value


def _sanitize_rows(rows):
    return [{k: _json_safe(v) for k, v in row.items()} for row in rows]


def _resolve_objids_file(objids_file: str | None) -> pathlib.Path | None:
    if not objids_file:
        return None
    path = pathlib.Path(objids_file)
    if not path.is_absolute():
        path = (DATA_DIR / path).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"objids file not found: {path}")
    return path


@app.get("/", response_class=HTMLResponse)
def index():
    return (TEMPLATES_DIR / "scanner.html").read_text()


@app.get("/api/objects")
def api_objects(objids_file: str | None = Query(default=None)):
    path = _resolve_objids_file(objids_file)
    if path is not None:
        with open(path) as f:
            objids = [line.strip() for line in f if line.strip()]
    else:
        with h5py.File(STATE["h5_path"], "r") as h5file:
            objids = sorted(k for k in h5file.keys() if k != "catalog")
    return {"objids": objids}


@app.get("/api/objects/{objid}/lightcurve")
def api_lightcurve(objid: str):
    with h5py.File(STATE["h5_path"], "r") as h5file:
        if objid not in h5file:
            raise HTTPException(status_code=404, detail=f"objid not found: {objid}")
        rows = load_object_rows(h5file, objid)
        title_suffix = crossmatch_title_suffix(h5file, objid)
        ra = float(h5file[objid].attrs["ra"])
        dec = float(h5file[objid].attrs["dec"])

    jd_first = first_detection_jd(rows) if rows else None

    return {
        "objid": objid,
        "ra": ra,
        "dec": dec,
        "title_suffix": title_suffix,
        "rows": _sanitize_rows(rows),
        "jd_first": jd_first,
        "jd_now": Time.now().jd,
    }


@app.get("/api/objects/{objid}/cutout/{epoch_idx}")
def api_cutout(objid: str, epoch_idx: int):
    with h5py.File(STATE["h5_path"], "r") as h5file:
        if objid not in h5file:
            raise HTTPException(status_code=404, detail=f"objid not found: {objid}")
        try:
            return get_object_cutouts(h5file, objid, epoch_idx)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/favorites")
def api_get_favorites():
    favorites_file = STATE["favorites_file"]
    if favorites_file.exists():
        with open(favorites_file) as f:
            return {"favorites": sorted({line.strip() for line in f if line.strip()})}
    return {"favorites": []}


@app.post("/api/favorites/{objid}")
def api_add_favorite(objid: str):
    favorites_file = STATE["favorites_file"]
    existing = set()
    if favorites_file.exists():
        with open(favorites_file) as f:
            existing = {line.strip() for line in f if line.strip()}

    if objid not in existing:
        favorites_file.parent.mkdir(parents=True, exist_ok=True)
        with open(favorites_file, "a") as f:
            f.write(objid + "\n")
        logger.info(f"Added to favorites: {objid} -> {favorites_file}")

    return {"favorited": True}


def main():
    parser = argparse.ArgumentParser(description="Run the tidez web scanner.")
    parser.add_argument("--h5", type=pathlib.Path, default=DEFAULT_H5,
                         help="Master H5 file to read from.")
    parser.add_argument("--favorites-file", type=pathlib.Path, default=DEFAULT_FAVORITES_FILE,
                         help="Text file to append favorited objids to.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    STATE["h5_path"] = args.h5
    STATE["favorites_file"] = args.favorites_file

    logger.info(f"Serving {args.h5} at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
