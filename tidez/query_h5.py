"""Read-side helpers for the tidez master H5 file: lookup by objid or cone search."""

import pathlib

import h5py
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


def get_object(h5_path: pathlib.Path, objid: str):
    """Return (lightcurve, cutouts) for a single object ID, or None if absent."""
    with h5py.File(h5_path, "r") as f:
        if objid not in f:
            return None
        group = f[objid]
        lightcurve = group["lightcurve"][:]
        cutouts = {key: group["cutouts"][key][:] for key in group["cutouts"].keys()}
        return lightcurve, cutouts


def cone_search(h5_path: pathlib.Path, ra_deg: float, dec_deg: float, radius_arcsec: float):
    """Return a list of objids within radius_arcsec of (ra_deg, dec_deg)."""
    with h5py.File(h5_path, "r") as f:
        catalog = f["catalog"][:]

    if len(catalog) == 0:
        return []

    target = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    catalog_coords = SkyCoord(catalog["ra"] * u.deg, catalog["dec"] * u.deg)
    sep = target.separation(catalog_coords)
    mask = sep < (radius_arcsec * u.arcsec)

    return [objid.decode("ascii") for objid in catalog["objid"][mask]]
