"""
Crossmatch every object already in the master H5 (see build_h5.py) against
ZTF and LSST alerts via the BOOM alert broker, and write the results back
into the same H5 file. This is a separate pass from the (future) filter
step: crossmatching is the slow, network-bound, flaky part, so it's cached
once and the filter can be iterated on cheaply afterwards.

BOOM's cone_search endpoint accepts multiple objects in one query
(`object_coordinates` is a dict of arbitrary-key -> [ra, dec]), so objects
are queried in batches of BATCH_SIZE rather than one HTTP request per
object. This cuts the request count by ~BATCH_SIZE and is far friendlier to
the API; a short sleep is inserted between batches to stay well under
whatever rate limit BOOM enforces.

For each object and each survey:
  - Batch cone search BOOM at (ra, dec) from the object's existing H5 attrs,
    alongside up to BATCH_SIZE-1 other objects in the same request.
  - Alerts are grouped by objectId (an object can have multiple prior
    alerts). The group whose position is nearest the DECam target position
    is treated as the match; its full alert history becomes the stored
    lightcurve.
  - Any other objectIds found within the search radius are kept as a
    lightweight list for future reference, without pulling their full data.

Written layout (added under each existing /<objid>/ group):
  /<objid>/crossmatch/                attrs: radius_arcsec
  /<objid>/crossmatch/ztf/            attrs: objectId, separation_arcsec, n_alerts
  /<objid>/crossmatch/ztf/lightcurve   compound dataset: jd, mag, magerr, limmag, band, detected
  /<objid>/crossmatch/ztf/other_objectids   dataset of strings
  /<objid>/crossmatch/lsst/           (same shape as ztf, if a match was found)

Resumable the same way as build_h5.py: an object's /crossmatch group is only
created after both surveys have been queried and its arrays built in
memory, so it's either absent or complete. Flushed to disk after every
batch. Re-running skips any objid that already has a /crossmatch group.
"""

import argparse
import pathlib
import time
import traceback

import h5py
import numpy as np
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

from tidez.logger import get_logger
from tidez.progress import append_progress, load_or_bootstrap_progress

logger = get_logger("crossmatch_boom")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_H5 = DATA_DIR / "tidez_master.h5"
DEFAULT_CROSSMATCH_LIST = DATA_DIR / "boom_crossmatches.txt"
DEFAULT_PROGRESS_FILE = DATA_DIR / "boom_progress.txt"
DEFAULT_RADIUS_ARCSEC = 5.0
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

BATCH_SIZE = 500
SLEEP_BETWEEN_BATCHES = 2.0  # seconds

BOOM_BASE = "https://api.kaboom.caltech.edu"
BOOM_AUTH_URL = f"{BOOM_BASE}/auth"
BOOM_CONE_URL = f"{BOOM_BASE}/queries/cone_search"

SURVEY_CATALOGS = {
    "ztf": "ZTF_alerts",
    "lsst": "LSST_alerts",
}

LIGHTCURVE_DTYPE = np.dtype(
    [
        ("jd", "f8"),
        ("mag", "f4"),
        ("magerr", "f4"),
        ("limmag", "f4"),
        ("band", "S1"),
        ("detected", "?"),
    ]
)


# ---------------------------------------------------------------------------
# BOOM auth + cone search
# ---------------------------------------------------------------------------

_session = requests.Session()
_token = None


def _login():
    import os

    global _token
    username = os.getenv("BOOM_USERNAME")
    password = os.getenv("BOOM_PASSWORD")
    if not username or not password:
        raise RuntimeError("Set BOOM_USERNAME and BOOM_PASSWORD environment variables.")
    resp = _session.post(
        BOOM_AUTH_URL,
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    _token = resp.json()["access_token"]
    return _token


def _auth_header(force_relogin=False):
    global _token
    if _token is None or force_relogin:
        _login()
    return {"Content-Type": "application/json", "Authorization": f"Bearer {_token}"}


def cone_search_batch(coords: dict, survey: str, radius_arcsec: float):
    """Cone search BOOM for many objects in one request.

    `coords` maps an arbitrary key (here, objid) to [ra, dec]; at most
    BATCH_SIZE entries should be passed at a time. Returns a dict mapping
    each key present in the response to its raw list of alert docs (keys
    with no matches are simply absent, matching BOOM's response shape).
    """
    catalog = SURVEY_CATALOGS[survey]
    payload = {
        "catalog_name": catalog,
        "object_coordinates": {key: [ra, dec] for key, (ra, dec) in coords.items()},
        "radius": radius_arcsec,
        "unit": "Arcseconds",
    }

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.post(
                BOOM_CONE_URL,
                headers=_auth_header(),
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 401:
                logger.warning("BOOM token expired/rejected, re-authenticating.")
                _auth_header(force_relogin=True)
                continue
            resp.raise_for_status()
            return resp.json().get("data", {})
        except requests.RequestException as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                f"BOOM batch cone search failed (attempt {attempt}/{MAX_RETRIES}) "
                f"for {survey} ({len(coords)} objects): {exc}. Retrying in {wait}s."
            )
            time.sleep(wait)

    raise RuntimeError(f"BOOM batch cone search failed after {MAX_RETRIES} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# Match selection
# ---------------------------------------------------------------------------

def _group_by_objectid(alerts):
    groups = {}
    for alert in alerts:
        groups.setdefault(alert["objectId"], []).append(alert)
    return groups


def _nearest_group(groups, ra: float, dec: float):
    """Return (nearest_objectid, separation_arcsec, other_objectids)."""
    target = SkyCoord(ra * u.deg, dec * u.deg)

    objectids = list(groups.keys())
    positions = SkyCoord(
        [groups[oid][0]["candidate"]["ra"] for oid in objectids] * u.deg,
        [groups[oid][0]["candidate"]["dec"] for oid in objectids] * u.deg,
    )
    seps = target.separation(positions).arcsec

    best_idx = int(np.argmin(seps))
    nearest_objid = objectids[best_idx]
    separation = float(seps[best_idx])
    other_objectids = [oid for oid in objectids if oid != nearest_objid]

    return nearest_objid, separation, other_objectids


def _build_lightcurve(alerts):
    lc = np.empty(len(alerts), dtype=LIGHTCURVE_DTYPE)
    for i, alert in enumerate(alerts):
        c = alert["candidate"]
        band = (c.get("band") or "")[:1]
        isdiffpos = bool(c.get("isdiffpos", False))
        is_negative = bool(c.get("isNegative", False))
        lc[i] = (
            c.get("jd", np.nan),
            c.get("magpsf", np.nan) if c.get("magpsf") is not None else np.nan,
            c.get("sigmapsf", np.nan) if c.get("sigmapsf") is not None else np.nan,
            c.get("diffmaglim", np.nan) if c.get("diffmaglim") is not None else np.nan,
            band.encode("ascii"),
            isdiffpos and not is_negative,
        )
    return lc


# ---------------------------------------------------------------------------
# H5 read/write
# ---------------------------------------------------------------------------

def write_survey_match(h5file, objid, ra, dec, survey, alerts):
    """Write one object's crossmatch/<survey> group from already-fetched alerts."""
    if not alerts:
        return

    groups = _group_by_objectid(alerts)
    nearest_objid, separation, other_objectids = _nearest_group(groups, ra, dec)
    lightcurve = _build_lightcurve(groups[nearest_objid])

    survey_group = h5file[objid]["crossmatch"].create_group(survey)
    survey_group.attrs["objectId"] = nearest_objid
    survey_group.attrs["separation_arcsec"] = separation
    survey_group.attrs["n_alerts"] = len(lightcurve)

    survey_group.create_dataset(
        "lightcurve", data=lightcurve, compression="gzip", compression_opts=4
    )
    survey_group.create_dataset(
        "other_objectids",
        data=np.array(other_objectids, dtype="S32"),
        compression="gzip",
        compression_opts=4,
    )


def crossmatch_batch(h5file, batch_objids, radius_arcsec):
    """Cone search one batch of objects against every survey (one BOOM
    request per survey, covering the whole batch) and write results.

    Returns (n_processed, n_failed, matched_objids). On a network failure
    for a survey, the *entire* batch is skipped for that call (no partial
    /crossmatch groups are created) so a re-run retries the whole batch
    cleanly, matching the per-object resumability of the old code.
    """
    coords = {
        objid: (float(h5file[objid].attrs["ra"]), float(h5file[objid].attrs["dec"]))
        for objid in batch_objids
    }

    alerts_by_survey = {}
    for survey in SURVEY_CATALOGS:
        logger.info(f"Requesting BOOM cone search: {survey} ({len(batch_objids)} objects) ...")
        try:
            alerts_by_survey[survey] = cone_search_batch(coords, survey, radius_arcsec)
        except Exception:
            logger.error(f"Failed to batch-crossmatch {len(batch_objids)} objects:\n{traceback.format_exc()}")
            return 0, len(batch_objids), []

    matched_objids = []
    for objid in batch_objids:
        ra, dec = coords[objid]
        cm_group = h5file[objid].create_group("crossmatch")
        cm_group.attrs["radius_arcsec"] = radius_arcsec
        any_match = False
        for survey in SURVEY_CATALOGS:
            alerts = alerts_by_survey[survey].get(objid, [])
            write_survey_match(h5file, objid, ra, dec, survey, alerts)
            any_match = any_match or bool(alerts)
        if any_match:
            matched_objids.append(objid)

    return len(batch_objids), 0, matched_objids


def run(
    h5_path: pathlib.Path,
    radius_arcsec: float,
    output_list: pathlib.Path,
    limit: int = None,
    progress_file: pathlib.Path = DEFAULT_PROGRESS_FILE,
):
    matched_objids = []
    logger.info(f"Opening {h5_path} ...")
    with h5py.File(h5_path, "a") as h5file:
        objids = [k for k in h5file.keys() if k != "catalog"]
        if limit is not None:
            objids = objids[:limit]

        already_done = load_or_bootstrap_progress(progress_file, h5file, objids, "crossmatch", logger)
        todo = [objid for objid in objids if objid not in already_done]
        logger.info(
            f"{len(objids)} objects in {h5_path}; "
            f"{len(already_done)} already crossmatched, will be skipped."
        )

        n_processed = 0
        n_skipped = len(already_done)
        n_failed = 0

        batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
        pbar = tqdm(batches, desc="Crossmatching with BOOM", unit="batch")
        for i, batch in enumerate(pbar, start=1):
            pbar.set_postfix_str(f"batch {i}/{len(batches)} ({n_processed}/{len(todo)} objects)")

            processed, failed, batch_matched = crossmatch_batch(h5file, batch, radius_arcsec)
            n_processed += processed
            n_failed += failed
            matched_objids.extend(batch_matched)

            h5file.flush()
            if processed > 0:
                append_progress(progress_file, batch)
            logger.info(
                f"Flushed after batch {i}/{len(batches)} "
                f"(processed={n_processed}, skipped={n_skipped}, failed={n_failed})"
            )

            if i < len(batches):
                time.sleep(SLEEP_BETWEEN_BATCHES)

    output_list.parent.mkdir(parents=True, exist_ok=True)
    with open(output_list, "a" if already_done else "w") as f:
        for objid in matched_objids:
            f.write(objid + "\n")
    with open(output_list) as f:
        total_matches = sum(1 for line in f if line.strip())

    logger.info(
        f"Done. processed={n_processed}, skipped={n_skipped}, failed={n_failed}, "
        f"total_objects={len(objids)}"
    )
    logger.info(
        f"Found {len(matched_objids)} new crossmatches this run ({total_matches} total). "
        f"Objids with crossmatches written to {output_list}"
    )


def main():
    parser = argparse.ArgumentParser(description="Crossmatch tidez master H5 objects against BOOM (ZTF/LSST).")
    parser.add_argument("--h5", type=pathlib.Path, default=DEFAULT_H5,
                         help="Master H5 file to read objects from and write crossmatches into.")
    parser.add_argument("--radius-arcsec", type=float, default=DEFAULT_RADIUS_ARCSEC,
                         help="Cone search radius in arcseconds.")
    parser.add_argument("--output-list", type=pathlib.Path, default=DEFAULT_CROSSMATCH_LIST,
                         help="Text file to write objids that have >=1 BOOM crossmatch to.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N objects (for testing).")
    parser.add_argument("--progress-file", type=pathlib.Path, default=DEFAULT_PROGRESS_FILE,
                         help="Checkpoint file tracking already-processed objids, for fast "
                              "resumability on large H5 files (default: data/boom_progress.txt).")
    args = parser.parse_args()

    run(args.h5, args.radius_arcsec, args.output_list, args.limit, args.progress_file)


if __name__ == "__main__":
    main()
