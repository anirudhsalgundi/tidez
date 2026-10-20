"""
Plot the combined DECam + ZTF + LSST lightcurve for one object in the
master H5 file (see scanning_setup.py).

X axis: "Days ago" (since today) on the bottom, "Days since first
detection" on the top (twin axis, same tick positions, different labels).
Y axis: magnitude (inverted, brighter = up). Detections are plotted as
points with error bars; non-detections are plotted as downward-pointing
open triangles at the limiting magnitude.
"""

import argparse
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from tidez.logger import get_logger

logger = get_logger("plot_lightcurve")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_H5 = DATA_DIR / "tidez_master.h5"

# Intuitive/mnemonic filter colors (u=violet/purple, g=green, r=red,
# i=orange, z=black, y=brown — 'y' has no strong convention requested, kept
# distinct from the rest). Shared across DECam/ZTF/LSST since the letters
# denote the same physical bands.
BAND_COLORS = {
    "u": "#8000ff",
    "g": "#2ca02c",
    "r": "#d62728",
    "i": "#ff7f0e",
    "z": "#000000",
    "y": "#8c564b",
}

MJD_TO_JD_OFFSET = 2400000.5

# Marker shape + a survey-specific edge (not just color) so surveys stay
# distinguishable even when the same band color overlaps.
SURVEY_STYLE = {
    "DECam": {"marker": "D", "edgecolor": "black",   "edgewidth": 1.0, "size": 55},
    "ZTF":   {"marker": "o", "edgecolor": "none",     "edgewidth": 0.0, "size": 55},
    "LSST":  {"marker": "^", "edgecolor": "#1f77b4",  "edgewidth": 1.3, "size": 65},
}


def _decam_rows(h5file, objid, good_only=True):
    """Every DECam epoch is kept either way -- good_only only changes which
    STATUS_FPHOT values count as a detection vs. an upper limit (plotted at
    LIM_MAG5, populated for every epoch regardless of status):
      good_only=True  (default): only 'p' (SNR>~5.8, good subtraction) counts
                       as detected; 'q' (marginal) and 'm' (non-detection)
                       both become upper limits.
      good_only=False: 'p' and 'q' both count as detected; only 'm' becomes
                       an upper limit.
    """
    rows = []
    lc = h5file[objid]["lightcurve"][:]
    for epoch_idx, row in enumerate(lc):
        band = row["FILTER"].decode("ascii").strip()
        if not band or band == "N/A":
            continue
        status = row["STATUS_FPHOT"].decode("ascii").strip()
        jd = float(row["MJD_OBS"]) + MJD_TO_JD_OFFSET
        detected_statuses = {"p"} if good_only else {"p", "q"}
        detected = status in detected_statuses
        rows.append({
            "jd": jd,
            "mag": float(row["MAG_FPHOT"]),
            "magerr": float(row["MAGERR_FPHOT"]),
            "limmag": float(row["LIM_MAG5"]),
            "band": band,
            "detected": detected,
            "survey": "DECam",
            "epoch_idx": epoch_idx,
            "snr": float(row["SNR_FPHOT"]),
            "status": status,
        })
    return rows


def _crossmatch_rows(h5file, objid, survey_key, survey_label):
    rows = []
    cm = h5file[objid].get("crossmatch")
    if cm is None or survey_key not in cm:
        return rows
    lc = cm[survey_key]["lightcurve"][:]
    for row in lc:
        band = row["band"].decode("ascii").strip()
        rows.append({
            "jd": float(row["jd"]),
            "mag": float(row["mag"]),
            "magerr": float(row["magerr"]),
            "limmag": float(row["limmag"]),
            "band": band or "?",
            "detected": bool(row["detected"]),
            "survey": survey_label,
            "epoch_idx": None,  # no cutouts stored for crossmatched surveys
            "snr": None,  # SNR/STATUS_FPHOT are DECam-specific (forced photometry on the diff image)
            "status": None,
        })
    return rows


def crossmatch_title_suffix(h5file, objid):
    """'  |  ZTF: <objectId>, LSST: <objectId>' for whichever surveys matched, else ''."""
    cm = h5file[objid].get("crossmatch")
    if cm is None:
        return ""
    parts = []
    for survey_key, survey_label in (("ztf", "ZTF"), ("lsst", "LSST")):
        if survey_key in cm:
            parts.append(f"{survey_label}: {cm[survey_key].attrs['objectId']}")
    return ("  |  " + ", ".join(parts)) if parts else ""


def load_object_rows(h5file, objid, good_only=True):
    """Every DECam epoch is always included; good_only (default True)
    controls only whether 'q' counts as a detection or an upper limit --
    see _decam_rows. ZTF/LSST crossmatch rows are always included as-is."""
    rows = _decam_rows(h5file, objid, good_only=good_only)
    rows += _crossmatch_rows(h5file, objid, "ztf", "ZTF")
    rows += _crossmatch_rows(h5file, objid, "lsst", "LSST")
    return rows


def first_detection_jd(rows):
    """jd of the first detection, or the earliest row's jd if nothing was detected."""
    detected_jds = [r["jd"] for r in rows if r["detected"]]
    return min(detected_jds) if detected_jds else min(r["jd"] for r in rows)


def render_lightcurve(ax, rows, objid, title_suffix="", title_extra=""):
    """Draw a combined lightcurve into `ax` (creates a twiny for the top axis).
    Returns (ax2, legend, artists_by_label) so callers can wire up interactivity.
    """
    legend_handles = {}
    artists_by_label = {}

    if not rows:
        ax.set_title(f"{objid}{title_suffix}  (no lightcurve data){title_extra}", fontsize=12, pad=20)
        return None, None, artists_by_label

    jd_now = Time.now().jd
    jd_first = first_detection_jd(rows)

    for r in rows:
        t = r["jd"] - jd_first
        color = BAND_COLORS.get(r["band"], "#888888")
        style = SURVEY_STYLE.get(r["survey"], {"marker": "x", "edgecolor": "none", "edgewidth": 0.0, "size": 50})

        if r["detected"]:
            label = f"{r['band']} ({r['survey']})"
            h = ax.errorbar(
                t, r["mag"], yerr=r["magerr"],
                fmt=style["marker"],
                markerfacecolor=color,
                markeredgecolor=style["edgecolor"] if style["edgecolor"] != "none" else color,
                markeredgewidth=style["edgewidth"] if style["edgewidth"] > 0 else 0.4,
                ecolor=color,
                markersize=7, capsize=2,
                linewidth=0, elinewidth=0.8,
                label=label, zorder=3, picker=5,
            )
        else:
            label = f"{r['band']} upper limit ({r['survey']})"
            h = ax.scatter(
                t, r["limmag"],
                marker="v",
                facecolors=color,
                edgecolors=style["edgecolor"] if style["edgecolor"] != "none" else color,
                linewidths=style["edgewidth"] if style["edgewidth"] > 0 else 0.4,
                s=style["size"] * 0.7, alpha=0.35,
                label=label, zorder=2, picker=5,
            )

        artists_by_label.setdefault(label, []).append(h)
        if label not in legend_handles:
            legend_handles[label] = h

    ax.invert_yaxis()
    ax.set_ylabel("Magnitude", fontsize=11)

    total_span = jd_now - jd_first

    def to_days_ago(x):
        return total_span - x

    ax.set_xlabel("Days ago", fontsize=11)
    bottom_ticks = ax.get_xticks()
    ax.set_xticks(bottom_ticks)
    ax.set_xticklabels([f"{to_days_ago(x):.0f}" for x in bottom_ticks])

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(bottom_ticks)
    ax2.set_xticklabels([f"{x:.0f}" for x in bottom_ticks])
    ax2.set_xlabel("Days since first detection", fontsize=11)

    ax.tick_params(axis="both", direction="in", which="both", top=False)
    ax2.tick_params(axis="both", direction="in", which="both")

    for spine in list(ax.spines.values()) + list(ax2.spines.values()):
        spine.set_linewidth(0.7)

    legend = ax.legend(
        handles=list(legend_handles.values()),
        labels=list(legend_handles.keys()),
        fontsize=8, framealpha=0.2,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    ax.set_title(f"{objid}{title_suffix}{title_extra}", fontsize=12, pad=20)

    return ax2, legend, artists_by_label


def plot_object_lightcurve(h5_path: pathlib.Path, objid: str, save_path: pathlib.Path = None):
    with h5py.File(h5_path, "r") as h5file:
        if objid not in h5file:
            raise KeyError(f"objid not found in {h5_path}: {objid}")

        rows = load_object_rows(h5file, objid)
        title_suffix = crossmatch_title_suffix(h5file, objid)

    if not rows:
        logger.warning(f"No lightcurve data found for {objid}")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    render_lightcurve(ax, rows, objid, title_suffix=title_suffix)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved lightcurve plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot the combined DECam+ZTF+LSST lightcurve for one object.")
    parser.add_argument("objid", type=str, help="Object ID to plot.")
    parser.add_argument("--h5", type=pathlib.Path, default=DEFAULT_H5,
                         help="Master H5 file to read from.")
    parser.add_argument("--save", type=pathlib.Path, default=None,
                         help="Save the plot to this path instead of showing it interactively.")
    args = parser.parse_args()

    plot_object_lightcurve(args.h5, args.objid, args.save)


if __name__ == "__main__":
    main()
