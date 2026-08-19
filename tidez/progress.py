"""
Shared resumability checkpointing for the crossmatch_*.py batch scripts.

Checking "is this object already done?" by querying the H5 file directly
(`"<group>" in h5file[objid]` for every object) is correct but doesn't
scale: each check is its own HDF5 metadata lookup, and at tens of
thousands of objects on a networked filesystem those add up to minutes of
silent, repeated work -- on *every* run, even ones that end up with
nothing left to do.

Instead, each script keeps a plain-text checkpoint file listing objids
already processed (whether or not they matched anything). Resuming is then
a single fast file read instead of one HDF5 lookup per object. The H5 file
remains the authoritative data store; the checkpoint file is purely a
resumability cache and is safe to delete (the next run just re-bootstraps
it from the H5, once, with the slow full scan as a fallback).
"""

import pathlib

from tqdm import tqdm


def load_or_bootstrap_progress(progress_file: pathlib.Path, h5file, objids, group_name: str, logger) -> set:
    """Return the set of objids already processed, from the checkpoint file
    if it exists, or by scanning the H5 file once (and writing the
    checkpoint file for next time) if it doesn't."""
    if progress_file.exists():
        with open(progress_file) as f:
            return {line.strip() for line in f if line.strip()}

    logger.info(f"No progress checkpoint at {progress_file}; scanning H5 once to bootstrap it.")
    done = {
        objid for objid in tqdm(objids, desc=f"Checking existing /{group_name} groups", unit="obj")
        if group_name in h5file[objid]
    }
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "w") as f:
        for objid in done:
            f.write(objid + "\n")
    return done


def append_progress(progress_file: pathlib.Path, objids):
    """Record more objids as done. Called after each object/chunk is
    successfully written to the H5, so the checkpoint file and the H5 file
    stay in sync even if the run is interrupted partway."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "a") as f:
        for objid in objids:
            f.write(objid + "\n")
