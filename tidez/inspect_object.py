"""
Print one object's full lightcurve as a plain-text table -- detections and
upper limits both, DECam+ZTF+LSST pooled -- for quick inspection without
needing a display (e.g. over SSH on a cluster).

Usage:
    python -m tidez.inspect_object <objid>
    python -m tidez.inspect_object <objid> --h5 data/tidez_master.h5
    python -m tidez.inspect_object <objid> --plot-all   # 'q' counts as detected too
"""

import argparse
import pathlib

import h5py

from tidez.plot_lightcurve import DEFAULT_H5, crossmatch_title_suffix, load_object_rows


def main():
    parser = argparse.ArgumentParser(description="Print one object's lightcurve rows as a table.")
    parser.add_argument("objid", type=str, help="Object ID to inspect.")
    parser.add_argument("--h5", type=pathlib.Path, default=DEFAULT_H5,
                         help="Master H5 file to read from.")
    parser.add_argument("--plot-all", action="store_true",
                         help="Count STATUS_FPHOT == 'q' as a detection too, not just 'p' "
                              "(default: only 'p' counts as detected; 'q'/'m' show as upper limits).")
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as h5file:
        if args.objid not in h5file:
            raise SystemExit(f"objid not found in {args.h5}: {args.objid}")

        ra = float(h5file[args.objid].attrs["ra"])
        dec = float(h5file[args.objid].attrs["dec"])
        title_suffix = crossmatch_title_suffix(h5file, args.objid)
        rows = load_object_rows(h5file, args.objid, good_only=not args.plot_all)

    print(f"{args.objid}{title_suffix}")
    print(f"ra={ra:.6f}  dec={dec:.6f}")
    print()

    rows = sorted(rows, key=lambda r: r["jd"])
    n_det = sum(1 for r in rows if r["detected"])
    n_ul = len(rows) - n_det
    print(f"{len(rows)} epochs total: {n_det} detections, {n_ul} upper limits")
    print()

    header = f"{'jd':>16}  {'survey':<6} {'band':<4} {'kind':<10} {'mag/limmag':>10}  {'magerr':>7}  {'snr':>6}  {'status':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        kind = "detection" if r["detected"] else "UPPER LIM"
        value = r["mag"] if r["detected"] else r["limmag"]
        magerr = f"{r['magerr']:.3f}" if r["detected"] and r["magerr"] is not None else "-"
        snr = f"{r['snr']:.1f}" if r["snr"] is not None else "-"
        status = r["status"] if r["status"] is not None else "-"
        print(f"{r['jd']:16.6f}  {r['survey']:<6} {r['band']:<4} {kind:<10} {value:10.3f}  {magerr:>7}  {snr:>6}  {status:>6}")


if __name__ == "__main__":
    main()
