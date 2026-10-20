"""
Basic contaminant flags shared by every transient filter (not a script).

Each is_* function takes the objids being filtered and returns a boolean Series
indexed by objid (True = contaminant). Thresholds are keyword arguments, so each
filter can use its own. basic_flags() builds all of them at once:

    from tidez.basic_filtering import basic_flags
    flags = basic_flags(objids, decam_lc)           # is_rock ... is_on_tns, clean
    clean_objids = flags.index[flags["clean"]]

Flags and their data:
    is_rock      fewer than min_detections detections (any survey), unless   light curve,
                 a galaxy is nearby                                       context/LSDR10
    is_star      Gaia match with significant parallax / proper motion,    context/Gaia_DR3,
                 or (optionally) a Legacy Survey point-source match        context/LSPSC
    is_variable  known variable star                                     context/VSX
    is_agn       known quasar / AGN                                       context/milliquas_v8
    is_on_tns    already reported to TNS (local TNS dump)                 data/tns_public_objects.csv

is_on_tns is a label, not a contaminant: basic_flags() leaves it out of `clean`
unless tns_is_contaminant=True.

Context files come from enrich.py (data/context/<catalog>.parquet: every match within
5", nearest first); the TNS dump from download_tns_objects.py.
"""

import csv
import pathlib

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MASTER_FILE = DATA_DIR / "all_objects.csv"
CROSSMATCHES_DIR = DATA_DIR / "crossmatches"
TNS_FILE = DATA_DIR / "tns_public_objects.csv"

ROCK_MIN_DETECTIONS = 2
ROCK_GALAXY_MAX_SEP_ARCSEC = 2.0
NOT_GALAXY_TYPES = ("PSF", "DUP")  # Legacy Survey morphological types that aren't galaxies
STAR_MAX_SEP_ARCSEC = 1.5
# 5 sigma, not 3: Gaia astrometry of galaxy nuclei gives spurious ~3-sigma parallaxes and
# proper motions (e.g. 0.4 mas/yr at 3.5 sigma), while real stars sit at >~15 sigma.
STAR_MIN_PARALLAX_SNR = 5.0
STAR_MIN_PM_SNR = 5.0
STAR_MIN_PS_SCORE = 0.9
VARIABLE_MAX_SEP_ARCSEC = 2.0
AGN_MAX_SEP_ARCSEC = 1.5
TNS_MAX_SEP_ARCSEC = 2.0
TNS_COLUMNS = ["tns_name", "tns_type", "tns_redshift", "tns_discoverydate", "tns_internal_names"]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def nearest_match(catalog: str, max_sep_arcsec: float, columns: list | None = None,
                  CROSSMATCHES_DIR: pathlib.Path = CROSSMATCHES_DIR) -> pd.DataFrame:
    """Nearest source of `catalog` within max_sep_arcsec, one row per matched objid
    (indexed by objid): sep_arcsec, n_matches, id, plus `columns` (default: all)."""
    path = CROSSMATCHES_DIR / f"{catalog}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run enrich.py first")
    lead = ["decam_objid", "rank", "n_matches", "sep_arcsec", "id"]
    df = pd.read_parquet(path, columns=None if columns is None else list(dict.fromkeys(lead + columns)))
    df = df[(df["rank"] == 0) & (df["sep_arcsec"] <= max_sep_arcsec)]
    return df.drop(columns="rank").set_index("decam_objid")


def load_positions(master: pathlib.Path = MASTER_FILE) -> pd.DataFrame:
    """ra/dec per objid from the master file (first row wins for duplicated objids)."""
    df = pd.read_csv(master, usecols=["objid", "ra", "dec"]).dropna(subset=["ra", "dec"])
    return df.drop_duplicates("objid").set_index("objid")


def load_tns(tns_file: pathlib.Path = TNS_FILE) -> pd.DataFrame:
    """TNS public-objects dump as a DataFrame with tns_name, tns_type, tns_redshift,
    tns_discoverydate, tns_internal_names, ra, dec.
    The dump starts with a timestamp line before the header, which is skipped."""
    with open(tns_file, newline="") as f:
        first = next(csv.reader(f))
    skip = 0 if "name" in [c.strip().lower() for c in first] else 1
    df = pd.read_csv(tns_file, skiprows=skip, low_memory=False)
    dec_col = "declination" if "declination" in df.columns else "dec"
    return pd.DataFrame({
        "tns_name": df["name_prefix"].fillna("").astype(str) + " " + df["name"].astype(str),
        "tns_type": df["type"] if "type" in df.columns else None,
        "tns_redshift": pd.to_numeric(df["redshift"], errors="coerce") if "redshift" in df.columns else np.nan,
        "tns_discoverydate": df["discoverydate"] if "discoverydate" in df.columns else None,
        "tns_internal_names": df["internal_names"] if "internal_names" in df.columns else None,
        "ra": pd.to_numeric(df["ra"], errors="coerce"),
        "dec": pd.to_numeric(df[dec_col], errors="coerce"),
    }).dropna(subset=["ra", "dec"])


def _flag(objids, matched_index) -> pd.Series:
    objids = pd.Index(objids, name="objid")
    return pd.Series(objids.isin(matched_index), index=objids)


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def is_rock(objids, lc: pd.DataFrame, min_detections: int = ROCK_MIN_DETECTIONS,
            galaxy_max_sep_arcsec: float = ROCK_GALAXY_MAX_SEP_ARCSEC,
            CROSSMATCHES_DIR: pathlib.Path = CROSSMATCHES_DIR) -> pd.Series:
    """True if the object has fewer than min_detections detections in `lc` (any survey) --
    a single detection followed by non-detections is an asteroid, an artifact or a one-off
    star flicker -- *unless* a galaxy lies within galaxy_max_sep_arcsec (a Legacy Survey DR10
    source whose type isn't PSF/DUP), where a lone detection could be a real fast transient.
    `lc` needs objid and detected (the DECam light curve, or a combined stream)."""
    objids = pd.Index(objids, name="objid")
    n_det = lc.loc[lc["detected"], "objid"].value_counts().reindex(objids, fill_value=0)
    ls = pd.read_parquet(CROSSMATCHES_DIR / "LSDR10.parquet", columns=["decam_objid", "sep_arcsec", "objtype"])
    near_galaxy = ls.loc[(ls["sep_arcsec"] <= galaxy_max_sep_arcsec)
                         & ~ls["objtype"].isin(NOT_GALAXY_TYPES), "decam_objid"].unique()
    return pd.Series((n_det.to_numpy() < min_detections) & ~objids.isin(near_galaxy), index=objids)


def is_star(objids, max_sep_arcsec: float = STAR_MAX_SEP_ARCSEC,
            min_parallax_snr: float = STAR_MIN_PARALLAX_SNR, min_pm_snr: float = STAR_MIN_PM_SNR,
            use_point_source: bool = False, min_ps_score: float = STAR_MIN_PS_SCORE,
            CROSSMATCHES_DIR: pathlib.Path = CROSSMATCHES_DIR) -> pd.Series:
    """True if the nearest Gaia DR3 source within max_sep_arcsec has a significant parallax
    (parallax / error >= min_parallax_snr) or proper motion (pm / error >= min_pm_snr).

    With use_point_source=True, also True for a Legacy Survey point-source-catalog match
    with score >= min_ps_score (catches faint stars Gaia misses, but can also catch compact
    galaxies, so it's off by default).
    """
    gaia = nearest_match("Gaia_DR3", max_sep_arcsec,
                         ["parallax", "parallax_error", "pm", "pmra", "pmdec", "pmra_error", "pmdec_error"],
                         CROSSMATCHES_DIR)
    parallax_snr = gaia["parallax"] / gaia["parallax_error"]
    # error on the total proper motion, propagated from its RA/Dec components
    pm_error = np.hypot(gaia["pmra"] * gaia["pmra_error"], gaia["pmdec"] * gaia["pmdec_error"]) / gaia["pm"]
    pm_snr = gaia["pm"] / pm_error
    stars = gaia.index[(parallax_snr >= min_parallax_snr) | (pm_snr >= min_pm_snr)]
    if use_point_source:
        ps = nearest_match("LSPSC", max_sep_arcsec, ["score"], CROSSMATCHES_DIR)
        stars = stars.union(ps.index[ps["score"] >= min_ps_score])
    return _flag(objids, stars)


def is_variable(objids, max_sep_arcsec: float = VARIABLE_MAX_SEP_ARCSEC,
                CROSSMATCHES_DIR: pathlib.Path = CROSSMATCHES_DIR) -> pd.Series:
    """True if a VSX variable star lies within max_sep_arcsec."""
    return _flag(objids, nearest_match("VSX", max_sep_arcsec, [], CROSSMATCHES_DIR).index)


def is_agn(objids, max_sep_arcsec: float = AGN_MAX_SEP_ARCSEC, types: str | None = None,
           CROSSMATCHES_DIR: pathlib.Path = CROSSMATCHES_DIR) -> pd.Series:
    """True if a Milliquas quasar/AGN lies within max_sep_arcsec. `types` optionally restricts
    to Milliquas classes containing any of these letters (e.g. "QA" for quasars and AGN)."""
    mq = nearest_match("milliquas_v8", max_sep_arcsec, ["objtype"], CROSSMATCHES_DIR)
    if types:
        mq = mq[mq["objtype"].fillna("").str.contains(f"[{types}]")]
    return _flag(objids, mq.index)


def tns_matches(objids, positions: pd.DataFrame | None = None, tns: pd.DataFrame | None = None,
                max_sep_arcsec: float = TNS_MAX_SEP_ARCSEC) -> pd.DataFrame:
    """Nearest TNS object within max_sep_arcsec for each objid that has one:
    tns_name, tns_type, tns_redshift, tns_discoverydate, tns_internal_names, tns_sep_arcsec
    (indexed by objid)."""
    positions = load_positions() if positions is None else positions
    tns = load_tns() if tns is None else tns
    pos = positions.reindex(pd.Index(objids, name="objid")).dropna(subset=["ra", "dec"])
    if pos.empty or tns.empty:
        return pd.DataFrame(columns=TNS_COLUMNS + ["tns_sep_arcsec"])
    targets = SkyCoord(pos["ra"].to_numpy() * u.deg, pos["dec"].to_numpy() * u.deg)
    catalog = SkyCoord(tns["ra"].to_numpy() * u.deg, tns["dec"].to_numpy() * u.deg)
    idx, sep, _ = targets.match_to_catalog_sky(catalog)
    sep = sep.arcsec
    ok = sep <= max_sep_arcsec
    hit = tns.iloc[idx[ok]][TNS_COLUMNS].reset_index(drop=True)
    hit["tns_sep_arcsec"] = sep[ok]
    hit.index = pd.Index(pos.index[ok], name="objid")
    return hit


def is_on_tns(objids, positions: pd.DataFrame | None = None, tns: pd.DataFrame | None = None,
              max_sep_arcsec: float = TNS_MAX_SEP_ARCSEC) -> pd.Series:
    """True if a TNS object lies within max_sep_arcsec (local TNS dump)."""
    return _flag(objids, tns_matches(objids, positions, tns, max_sep_arcsec).index)


# ---------------------------------------------------------------------------
# All together
# ---------------------------------------------------------------------------

def basic_flags(objids, lc: pd.DataFrame, tns_is_contaminant: bool = False,
                rock_kwargs: dict | None = None, star_kwargs: dict | None = None,
                variable_kwargs: dict | None = None, agn_kwargs: dict | None = None,
                tns_kwargs: dict | None = None) -> pd.DataFrame:
    """DataFrame indexed by objid with is_rock, is_star, is_variable, is_agn, is_on_tns, the
    TNS name/type/separation where matched, and `clean` = none of the contaminant flags set.
    Per-flag thresholds go in the *_kwargs dicts, e.g. star_kwargs={"max_sep_arcsec": 1.0}."""
    objids = pd.Index(pd.unique(np.asarray(objids)), name="objid")
    tns_kwargs = tns_kwargs or {}
    tns_hits = tns_matches(objids, **tns_kwargs)
    flags = pd.DataFrame({
        "is_rock": is_rock(objids, lc, **(rock_kwargs or {})),
        "is_star": is_star(objids, **(star_kwargs or {})),
        "is_variable": is_variable(objids, **(variable_kwargs or {})),
        "is_agn": is_agn(objids, **(agn_kwargs or {})),
        "is_on_tns": objids.isin(tns_hits.index),
    }, index=objids)
    contaminants = ["is_rock", "is_star", "is_variable", "is_agn"] + (["is_on_tns"] if tns_is_contaminant else [])
    flags["clean"] = ~flags[contaminants].any(axis=1)
    return flags.join(tns_hits)


def flag_summary(flags: pd.DataFrame) -> pd.Series:
    """How many objects each flag catches, plus how many are clean -- for logging the funnel."""
    cols = [c for c in flags.columns if c.startswith("is_")] + ["clean"]
    return flags[cols].sum().astype(int).rename("n_objects")
