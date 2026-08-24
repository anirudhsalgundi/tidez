"""
Dump photometry (mjd, filter, mag, mag_err, upper_lim, instrument, ...) for
one or more objects from the master H5 file (see build_h5.py, plot_lightcurve.py).

Usage:
    python -m tidez.get_photometry <objid>
    python -m tidez.get_photometry <objid1> <objid2> ...
    python -m tidez.get_photometry --objid-file objids.txt
    python -m tidez.get_photometry <objid> --plot-all   # 'q' counts as detected too
    python -m tidez.get_photometry <objid> --out photometry.csv
    python -m tidez.get_photometry --objid-file objids.txt --out-dir photometry/

Columns:
    objid, mjd, filter, mag, mag_err, upper_lim, limmag, snr, instrument, status
        mjd        - Modified Julian Date of the observation
        filter     - photometric band (u/g/r/i/z/y)
        mag        - magnitude (only meaningful when upper_lim is False)
        mag_err    - magnitude error (only meaningful when upper_lim is False)
        upper_lim  - True if this epoch is a non-detection (mag/mag_err are NaN)
        limmag     - 5-sigma limiting magnitude for this epoch
        snr        - signal-to-noise ratio (DECam forced photometry only, else NaN)
        instrument - survey/instrument name (DECam, ZTF, LSST)
        status     - STATUS_FPHOT code (DECam only, else empty)
"""

import argparse
import csv
import pathlib
import sys

import h5py

from tidez.plot_lightcurve import DEFAULT_H5, MJD_TO_JD_OFFSET, load_object_rows

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

COLUMNS = ["objid", "mjd", "filter", "mag", "mag_err", "upper_lim",
           "limmag", "snr", "instrument", "status"]


def get_photometry(h5file: h5py.File, objid: str, good_only: bool = True) -> list[dict]:
    """Return a list of photometry row dicts for a single object."""
    if objid not in h5file:
        raise KeyError(f"objid not found in H5 file: {objid}")

    rows = load_object_rows(h5file, objid, good_only=good_only)
    records = []
    for r in sorted(rows, key=lambda r: r["jd"]):
        records.append({
            "objid": objid,
            "mjd": r["jd"] - MJD_TO_JD_OFFSET,
            "filter": r["band"],
            "mag": r["mag"] if r["detected"] else "",
            "mag_err": r["magerr"] if r["detected"] else "",
            "upper_lim": r["detected"] is False,
            "limmag": r["limmag"],
            "snr": r["snr"] if r["snr"] is not None else "",
            "instrument": r["survey"],
            "status": r["status"] if r["status"] is not None else "",
        })
    return records


def read_objid_file(path: pathlib.Path) -> list[str]:
    objids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        objids.append(line.split()[0])
    return objids


def main():
    parser = argparse.ArgumentParser(description="Get photometry table(s) for object(s) from the master H5 file.")
    parser.add_argument("objid", type=str, nargs="*", help="Object ID(s) to fetch photometry for.")
    parser.add_argument("--objid-file", type=pathlib.Path, default=None,
                         help="Text file with one objid per line (in addition to any given positionally).")
    parser.add_argument("--h5", type=pathlib.Path, default=DEFAULT_H5,
                         help="Master H5 file to read from.")
    parser.add_argument("--plot-all", action="store_true",
                         help="Count STATUS_FPHOT == 'q' as a detection too, not just 'p' "
                              "(default: only 'p' counts as detected; 'q'/'m' show as upper limits).")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                         help="Write a single objid's photometry to this CSV path instead of stdout.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None,
                         help="Write one CSV per object (named <objid>.csv) into this directory.")
    args = parser.parse_args()

    objids = list(args.objid)
    if args.objid_file is not None:
        objids += read_objid_file(args.objid_file)
    if not objids:
        parser.error("no objids given: pass them positionally or via --objid-file")
    if args.out is not None and len(objids) > 1:
        parser.error("--out only supports a single objid; use --out-dir for multiple")

    with h5py.File(args.h5, "r") as h5file:
        tables = {}
        for objid in objids:
            try:
                tables[objid] = get_photometry(h5file, objid, good_only=not args.plot_all)
            except KeyError as e:
                print(f"warning: {e}", file=sys.stderr)

    if not tables:
        raise SystemExit("no valid objids found")

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for objid, rows in tables.items():
            out_path = args.out_dir / f"{objid}.csv"
            _write_csv(out_path, rows)
            print(f"wrote {out_path}")
        return

    combined_rows = [row for rows in tables.values() for row in rows]

    if args.out is not None:
        _write_csv(args.out, combined_rows)
        print(f"wrote {args.out}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(combined_rows)


def _write_csv(path: pathlib.Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
