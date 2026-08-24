"""
Build a subset H5 file containing only the objects listed in a text file.

Copies each objid's whole group (attrs, lightcurve, cutouts, and any
optional subgroups like legacy_survey) verbatim from the source H5 into
a new output H5. Objects missing optional subgroups (e.g. no legacy_survey
crossmatch yet) are copied as-is -- nothing is filled in or required.

Usage:
    python -m tidez.subset_h5 favorite_sources.txt --out favorites.h5
    python -m tidez.subset_h5 sne_sources.txt --out sne.h5 --h5 data/tidez_master.h5
"""

import argparse
import pathlib
import sys

import h5py

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_H5 = DATA_DIR / "tidez_master.h5"


def read_objid_file(path: pathlib.Path) -> list[str]:
    objids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        objids.append(line.split()[0])
    return objids


def build_subset(src_path: pathlib.Path, objids: list[str], out_path: pathlib.Path):
    n_copied = 0
    n_missing = 0
    with h5py.File(src_path, "r") as src, h5py.File(out_path, "w") as dst:
        for objid in objids:
            if objid not in src:
                print(f"warning: objid not found in {src_path}: {objid}", file=sys.stderr)
                n_missing += 1
                continue
            if objid in dst:
                continue
            src.copy(src[objid], dst, name=objid)
            n_copied += 1
    return n_copied, n_missing


def main():
    parser = argparse.ArgumentParser(
        description="Build a subset H5 file from a text file of objids.")
    parser.add_argument("objid_file", type=pathlib.Path,
                         help="Text file with one objid per line.")
    parser.add_argument("--out", type=pathlib.Path, required=True,
                         help="Path to the output H5 file (created/overwritten).")
    parser.add_argument("--h5", type=pathlib.Path, default=DEFAULT_H5,
                         help="Master H5 file to read from.")
    args = parser.parse_args()

    objids = read_objid_file(args.objid_file)
    if not objids:
        parser.error(f"no objids found in {args.objid_file}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_copied, n_missing = build_subset(args.h5, objids, args.out)
    print(f"wrote {args.out}: {n_copied} objects copied, {n_missing} missing from {args.h5}")


if __name__ == "__main__":
    main()
