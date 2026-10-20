"""
Web-based interactive scanner for the tidez master H5.

Serves an interactive lightcurve (Plotly) for each object; clicking a DECam
data point loads its template/science/diff cutouts below. Prev/Next step
through objects, an objid-file field restricts the browsing list. The TNS
button opens the matched TNS page, with a badge saying whether our DECam
objid is among its internal names. Favorite and To-report toggle the object
in data/favorite_sources.txt and data/tns_to_report.txt.

Usage:
    python -m tidez.webapp
    python -m tidez.webapp --h5 data/tidez_master.h5 --port 8000
"""

import argparse
import csv
import functools
import pathlib

import math

import h5py
import uvicorn
from astropy.time import Time
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from tidez.cutouts import get_object_cutouts
from tidez.legacy_cutout import get_object_legacy_cutout
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
DEFAULT_REPORT_FILE = DATA_DIR / "tns_to_report.txt"
MASTER_FILE = DATA_DIR / "all_objects.csv"

app = FastAPI(title="tidez scanner")


# Browsers will happily cache GET /api/... responses (no explicit headers
# say otherwise), which silently serves stale lightcurve/objects data across
# server restarts (e.g. toggling --good-only). Force no-store on every
# response so a restart always takes effect immediately.
@app.middleware("http")
async def no_cache(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response

# Set by main() before uvicorn.run(); read by the request handlers below.
STATE = {"h5_path": DEFAULT_H5, "good_only": True,
         "lists": {"favorites": DEFAULT_FAVORITES_FILE, "report": DEFAULT_REPORT_FILE}}


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


@functools.lru_cache(maxsize=1)
def _programs_from_master() -> dict:
    """objid -> program(s) from the master file, for H5 files without a 'program' attr.
    An objid found under more than one program gets them comma-joined."""
    if not MASTER_FILE.exists():
        return {}
    programs = {}
    with open(MASTER_FILE, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("program"):
                programs.setdefault(row["objid"], []).append(row["program"])
    return {objid: ", ".join(dict.fromkeys(p)) for objid, p in programs.items()}


def _object_program(group, objid: str) -> str | None:
    program = group.attrs.get("program")
    if program is not None:
        return program.decode() if isinstance(program, bytes) else str(program)
    return _programs_from_master().get(objid)


def _text_attr(group, key: str) -> str | None:
    """A text attr from the scan bundle; "" / NaN / missing -> None."""
    value = group.attrs.get(key)
    if isinstance(value, bytes):
        value = value.decode()
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == "":
        return None
    return str(value).strip()


def _tns_info(group) -> dict:
    """TNS match stored by make_scan_bundle.py. tns_url uses the bare name
    ("AT 2025abc" -> https://www.wis-tns.org/object/2025abc)."""
    tns_name = _text_attr(group, "tns_name")
    decam_on_tns = group.attrs.get("decam_on_tns")
    return {
        "tns_name": tns_name,
        "tns_type": _text_attr(group, "tns_type"),
        "tns_url": f"https://www.wis-tns.org/object/{tns_name.split()[-1]}" if tns_name else None,
        "decam_on_tns": bool(decam_on_tns) if decam_on_tns is not None else None,
    }


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
            if "scan_order" in h5file:  # scan bundles: the filter's ranking, fastest first
                objids = [o.decode() if isinstance(o, bytes) else str(o) for o in h5file["scan_order"][:]]
            else:
                objids = sorted(k for k in h5file.keys() if isinstance(h5file[k], h5py.Group))
    return {"objids": objids}


@app.get("/api/objects/{objid}/lightcurve")
def api_lightcurve(objid: str):
    with h5py.File(STATE["h5_path"], "r") as h5file:
        if objid not in h5file:
            raise HTTPException(status_code=404, detail=f"objid not found: {objid}")
        rows = load_object_rows(h5file, objid, good_only=STATE["good_only"])
        title_suffix = crossmatch_title_suffix(h5file, objid)
        ra = float(h5file[objid].attrs["ra"])
        dec = float(h5file[objid].attrs["dec"])
        program = _object_program(h5file[objid], objid)
        tns = _tns_info(h5file[objid])
        cm = h5file[objid].get("crossmatch")
        ztf_id = str(cm["ztf"].attrs["objectId"]) if cm is not None and "ztf" in cm else None
        lsst_id = str(cm["lsst"].attrs["objectId"]) if cm is not None and "lsst" in cm else None

    jd_first = first_detection_jd(rows) if rows else None

    return {
        "objid": objid,
        "ra": ra,
        "dec": dec,
        "program": program,
        **tns,
        "ztf_id": ztf_id,
        "lsst_id": lsst_id,
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


@app.get("/api/objects/{objid}/legacy-cutout")
def api_legacy_cutout(objid: str):
    with h5py.File(STATE["h5_path"], "r") as h5file:
        if objid not in h5file:
            raise HTTPException(status_code=404, detail=f"objid not found: {objid}")
        cutout = get_object_legacy_cutout(h5file, objid)
        if cutout is None:
            raise HTTPException(status_code=404, detail=f"no Legacy Survey cutout stored for {objid}")
        return cutout


def _list_file(name: str) -> pathlib.Path:
    if name not in STATE["lists"]:
        raise HTTPException(status_code=404, detail=f"unknown list: {name}")
    return STATE["lists"][name]


def _read_list(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return list(dict.fromkeys(line.strip() for line in f if line.strip()))


@app.get("/api/lists/{name}")
def api_get_list(name: str):
    """Objids in a list: favorites (favorite_sources.txt) or report (tns_to_report.txt)."""
    return {"objids": _read_list(_list_file(name))}


@app.post("/api/lists/{name}/{objid}")
def api_add_to_list(name: str, objid: str):
    path = _list_file(name)
    if objid not in _read_list(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(objid + "\n")
        logger.info(f"Added to {name}: {objid} -> {path}")
    return {"in_list": True}


@app.delete("/api/lists/{name}/{objid}")
def api_remove_from_list(name: str, objid: str):
    path = _list_file(name)
    objids = _read_list(path)
    if objid in objids:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(f"{o}\n" for o in objids if o != objid))
        tmp.replace(path)
        logger.info(f"Removed from {name}: {objid} -> {path}")
    return {"in_list": False}


def main():
    parser = argparse.ArgumentParser(description="Run the tidez web scanner.")
    parser.add_argument("--h5", type=pathlib.Path, default=DEFAULT_H5,
                         help="Master H5 file to read from.")
    parser.add_argument("--favorites-file", type=pathlib.Path, default=DEFAULT_FAVORITES_FILE,
                         help="Text file to append favorited objids to.")
    parser.add_argument("--report-file", type=pathlib.Path, default=DEFAULT_REPORT_FILE,
                         help="Text file of objids whose DECam data still needs reporting to TNS.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--plot-all", action="store_true",
                         help="Also count STATUS_FPHOT == 'q' (marginal) DECam epochs as "
                              "detections, not just 'p' (good subtraction). Every DECam epoch "
                              "is plotted either way ('m'/non-detection epochs always show as "
                              "upper limits at LIM_MAG5) -- this flag only moves the boundary "
                              "between 'plotted as a detection' and 'plotted as an upper limit'. "
                              "ZTF/LSST crossmatch points are always shown when present either way.")
    args = parser.parse_args()

    STATE["h5_path"] = args.h5
    STATE["lists"] = {"favorites": args.favorites_file, "report": args.report_file}
    STATE["good_only"] = not args.plot_all

    detection_mode = "DECam: 'p'+'q' count as detections" if args.plot_all else "DECam: only 'p' counts as detected, good-only"
    logger.info(f"Serving {args.h5} at http://{args.host}:{args.port} ({detection_mode})")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()