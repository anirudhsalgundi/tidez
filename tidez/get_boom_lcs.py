"""
Step 3: fetch ZTF and LSST light curves from BOOM for every object in the master file.

For each batch of objects in data/all_objects.csv, one cone search per survey on
BOOM's *_alerts_aux catalog returns each matched object's full history in one document:

  prv_candidates     alert detections           kind = "alert"
                     (negative subtractions)    kind = "alert_negative"
  prv_nondetections  ZTF upper limits           kind = "nondetection"
  fp_hists           forced photometry          kind = "fp"

The nearest BOOM object within --radius-arcsec is the match. Rows follow the DECam
light-curve convention: detected = True rows carry mag/magerr; every other row is an
upper limit at limmag (= diffmaglim) with mag = magerr = NaN. Forced photometry counts
as a detection at SNR >= 5 with a positive difference (same 5-sigma bar as DECam);
ZTF forced photometry with procstatus != 0 is dropped.

Output: data/boom_lc.parquet, one row per ZTF/LSST measurement:
    decam_objid, ztfname, lsstname, survey, kind, mjd, filter, mag, magerr, limmag,
    snr, detected, ztf_sep_arcsec, lsst_sep_arcsec
ztfname/lsstname (and their separations) are the object's matches, repeated on every
row; survey says which of the two a row's measurement came from. Objects with no
match in either survey have no rows.

Resumable: each batch is saved to data/boom_lc_parts/ as it finishes, and a re-run
skips finished batches. Once every object is done, the parts are merged into
boom_lc.parquet and the parts directory is removed (so the next run starts fresh,
picking up new alerts).

Needs BOOM_USERNAME and BOOM_PASSWORD in the environment.

Usage:
    python tidez/get_boom_lcs.py
    python tidez/get_boom_lcs.py --radius-arcsec 3
"""

import argparse
import csv
import json
import math
import os
import pathlib
import re
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm
from boom import BoomClient
import pandas as pd
import json

from tidez.logger import get_logger
logger = get_logger("get_boom_lcs")

client = BoomClient(username=os.environ.get("BOOM_USERNAME"), password=os.environ.get("BOOM_PASSWORD", "..."))

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MASTER_CSV = DATA_DIR / "all_objects.csv"
MASTER_PARAQUET = DATA_DIR / "all_objects.parquet"

CHUNK_SIZE = 500
SEARCH_RADIUS = 2
BOOM_CATALOGS = ["ZTF_alerts", "LSST_alerts"]



def boom_init():
    from boom import BoomClient
    boom = BoomClient(username=os.environ.get("BOOM_USERNAME"), password=os.environ.get("BOOM_PASSWORD"))

    return boom


def create_boom_targets(lightcurve_file: pathlib.Path) -> list:
    """Return a list of dicts with 'ra', 'dec', and 'radius_arcsec' for each object in the lightcurve file."""
    df = pd.read_parquet(lightcurve_file, columns=["objid", "ra", "dec"])
    objid, ra, dec = df["objid"].tolist(), df["ra"].tolist(), df["dec"].tolist()
    boom_targets = {objid[i]: (ra[i], dec[i]) for i in range(len(objid))}

    return boom_targets

def boom_cone_search(boom, boom_targets: dict) -> dict:

    target_names = list(boom_targets.keys())
    results = {catalog: {} for catalog in BOOM_CATALOGS}
    failed_chunks = []

    for i in tqdm.tqdm(range(0, len(target_names), CHUNK_SIZE), desc="BOOM cone searches"):
        chunk_names = target_names[i:i + CHUNK_SIZE]
        chunk_targets = {k: boom_targets[k] for k in chunk_names}

        for catalog in BOOM_CATALOGS:

            # in case the query breaks, try twice before giving up
            for attempt in range(2):
                try:
                    r = boom.cone_search(
                        catalog, chunk_targets, SEARCH_RADIUS, unit="arcsec",
                        filter=None,
                        projection={"objectId": 1},
                    )
                    results[catalog].update(r)

                    # if a query is successful, no need to retry
                    break

                # if it breaks
                except Exception as e:
                    # and it was first try
                    if attempt == 0:
                        #wait for 5 secsons and retry
                        time.sleep(5)
                    # if it fails even after that, add it as a failed query
                    else:
                        failed_chunks.append((catalog, i, str(e)))

    for catalog in BOOM_CATALOGS:
        n_matched = sum(1 for v in results[catalog].values() if v)
        print(f"{catalog}: {n_matched} of {len(target_names)} targets have at least one match")
    print(f"{len(failed_chunks)} chunk queries failed")

    json.dump(results, open(DATA_DIR / "boom_crossmatch_results.json", "w"), indent=2)

    return None


# using the boom cone search results, we need to build lightcurves for each object. But before that, if there is more than one ztf or lsst crossmatch, we need to pick the closest one.


def update_crossmatches(decam_target_coords):
    with open(DATA_DIR / "boom_crossmatch_results.json") as f:
        crossmatches = json.load(f)

    nearest = {"ZTF_alerts": {}, "LSST_alerts": {}}

    for survey in ["ZTF_alerts", "LSST_alerts"]:
        for objid, matches in crossmatches[survey].items():
            if len(matches) == 0 or objid not in decam_target_coords:
                continue

            decam_ra, decam_dec = decam_target_coords[objid]
            decam_coord = SkyCoord(ra=decam_ra * u.deg, dec=decam_dec * u.deg)

            alert_ra = np.array([m["candidate"]["ra"] for m in matches])
            alert_dec = np.array([m["candidate"]["dec"] for m in matches])
            alert_coords = SkyCoord(ra=alert_ra * u.deg, dec=alert_dec * u.deg)

            sep_arcsec = decam_coord.separation(alert_coords).arcsec
            i_best = int(np.argmin(sep_arcsec))

            best = dict(matches[i_best])
            best["separation"] = float(sep_arcsec[i_best])
            best["n_alerts"] = len(matches)
            nearest[survey][objid] = best

    print(f"ZTF targets with a match: {len(nearest['ZTF_alerts'])}")
    print(f"LSST targets with a match: {len(nearest['LSST_alerts'])}")

    nearest_df = pd.DataFrame({
        "objid": list(nearest["ZTF_alerts"].keys()) + list(nearest["LSST_alerts"].keys()),
        "ztfname": [nearest["ZTF_alerts"].get(objid, {}).get("objectId") for objid in nearest["ZTF_alerts"].keys()] +
                   [None] * len(nearest["LSST_alerts"]),
        "ztf_sep_arcsec": [nearest["ZTF_alerts"].get(objid, {}).get("separation") for objid in nearest["ZTF_alerts"].keys()] +
                          [None] * len(nearest["LSST_alerts"]),
        "lsstname": [None] * len(nearest["ZTF_alerts"]) +
                    [nearest["LSST_alerts"].get(objid, {}).get("objectId") for objid in nearest["LSST_alerts"].keys()],
        "lsst_sep_arcsec": [None] * len(nearest["ZTF_alerts"]) +
                           [nearest["LSST_alerts"].get(objid, {}).get("separation") for objid in nearest["LSST_alerts"].keys()],
    })

    # write this to the master paraquet file with columns, ztf name, lsstname. 
    return nearest




def main():
    parser = argparse.ArgumentParser(description="Fetch ZTF and LSST light curves from BOOM for every object in the master file.")
    parser.add_argument("--radius-arcsec", type=float, default=SEARCH_RADIUS, help="Search radius in arcseconds (default: 2)")
    args = parser.parse_args()

    boom = boom_init()
    boom_targets = create_boom_targets(MASTER_FILE)
    boom_cone_search(boom, boom_targets)

if __name__ == "__main__":
    main()