"""
Quick visual scan of a list of objects: combined DECam + ZTF + LSST light curve, DECam
cutouts, and one scrollable scanner.html page.

Reads (nothing is modified):
    data/all_objects.parquet         master: ra, dec, ztfname, lsstname
    data/decam_lightcurves.parquet   build_lcs.py: DECam light curve + brightest-point cutouts
    data/boom_lightcurves.parquet    get_boom_lcs.py: ZTF and LSST light curves

Light curve: ZTF circles, DECam squares, LSST (Rubin) diamonds, coloured by filter. Upper
limits are open downward triangles in the filter colour, drawn fainter. Which points are
drawn as upper limits is a display choice only (see is_drawn_as_detection); every point in
the files is drawn one way or the other, except points with no magnitude and no limit.

Cutouts: science / template / difference of the DECam brightest point, inferno colour map.

Outputs (for me by default, or for a collaborator with --collaborator NAME):
    exports/me/                          or exports/collaborators/<NAME>/
        scanner.html                     every object, one below the other
        plots/lightcurves/<objid>_lightcurve.png
        plots/cutouts/<objid>_cutouts.png
The folder is self-contained (scanner.html uses relative paths), so it can be zipped and sent.

Usage:
    python -m tidez.quick_scan objects.txt
    python -m tidez.quick_scan objects.txt --collaborator alice
objects.txt: one objid per line; blank lines and lines starting with # are ignored.
"""

import argparse
import html
import pathlib
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tqdm import tqdm

from tidez.logger import get_logger
logger = get_logger("quick_scan")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPORTS_DIR = PROJECT_ROOT / "exports"
MASTER_FILE = DATA_DIR / "all_objects.parquet"
DECAM_LIGHTCURVES_FILE = DATA_DIR / "decam_lightcurves.parquet"
BOOM_LIGHTCURVES_FILE = DATA_DIR / "boom_lightcurves.parquet"

# Display rule only (nothing is removed): a point is drawn as a detection when it has a
# magnitude, SNR >= this, and is not a negative subtraction; otherwise as an upper limit
# at its limiting magnitude.
DETECTION_SNR_FOR_PLOT = 5.0

# same filter colours as plot_lightcurve.py
FILTER_COLORS = {"u": "#8000ff", "g": "#2ca02c", "r": "#d62728", "i": "#ff7f0e", "z": "#000000", "y": "#8c564b"}
OTHER_FILTER_COLOR = "#888888"  # e.g. DECam 'N/A'
SURVEY_MARKERS = {"ZTF": "o", "DECam": "s", "LSST": "D"}
UPPER_LIMIT_MARKER = "v"
UPPER_LIMIT_ALPHA = 0.35
CUTOUT_SIDE = 121
CUTOUT_NAMES = {"cutout_science": "Science", "cutout_template": "Template", "cutout_difference": "Difference"}
BOOM_LIGHTCURVE_FIELDS = ["mjds", "filters", "mags", "magerrs", "limiting_mags", "snr_psfs", "isdiffpos", "kinds"]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def read_objid_list(path: pathlib.Path) -> list:
    objids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and line not in objids:
            objids.append(line)
    logger.info(f"{len(objids)} objects in {path}")
    return objids


def read_rows_for(path: pathlib.Path, objid_column: str, objids: list, columns: list | None = None) -> pd.DataFrame:
    """Only the rows of `path` for these objids (the light-curve files are too big to load whole)."""
    if not path.exists():
        logger.warning(f"{path} not found, skipping it")
        return pd.DataFrame(columns=[objid_column])
    rows = pd.read_parquet(path, columns=columns, filters=[(objid_column, "in", objids)])
    logger.info(f"Read {len(rows)} rows for {rows[objid_column].nunique()} of the objects from {path.name}")
    return rows


def first_row_per_objid(rows: pd.DataFrame, objid_column: str, label: str) -> dict:
    """{objid: row}; an objid listed under several programs keeps its first row (with a warning)."""
    duplicated = rows[rows[objid_column].duplicated()][objid_column].unique()
    for objid in duplicated:
        logger.warning(f"{objid} has {int((rows[objid_column] == objid).sum())} rows in {label}; using the first")
    return {objid: row for objid, row in rows.drop_duplicates(objid_column).set_index(objid_column, drop=False).iterrows()}


def output_dir_for(collaborator: str | None) -> pathlib.Path:
    if not collaborator:
        return EXPORTS_DIR / "me"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", collaborator.strip())
    return EXPORTS_DIR / "collaborators" / safe_name


# ---------------------------------------------------------------------------
# Light curve
# ---------------------------------------------------------------------------

def is_drawn_as_detection(mag, snr, isdiffpos=True) -> bool:
    return bool(np.isfinite(mag) and np.isfinite(snr) and snr >= DETECTION_SNR_FOR_PLOT and isdiffpos is not False)


def decam_points(decam_row) -> pd.DataFrame:
    points = pd.DataFrame({
        "mjd": decam_row["mjds"], "filter": decam_row["filters"], "mag": decam_row["mags"],
        "magerr": decam_row["magerrs"], "limiting_mag": decam_row["limiting_mags"], "snr": decam_row["snrs"],
    }).astype({"mjd": float, "mag": float, "magerr": float, "limiting_mag": float, "snr": float})
    points["detected"] = [is_drawn_as_detection(m, s) for m, s in zip(points["mag"], points["snr"])]
    points["survey"] = "DECam"
    return points


def boom_points(boom_row, survey_prefix: str, survey_label: str) -> pd.DataFrame:
    points = pd.DataFrame({field: list(boom_row[f"{survey_prefix}_{field}"]) for field in BOOM_LIGHTCURVE_FIELDS})
    points = points.rename(columns={"mjds": "mjd", "filters": "filter", "mags": "mag", "magerrs": "magerr",
                                    "limiting_mags": "limiting_mag", "snr_psfs": "snr"})
    points = points.astype({"mjd": float, "mag": float, "magerr": float, "limiting_mag": float, "snr": float})
    points["detected"] = [is_drawn_as_detection(m, s, p) for m, s, p in zip(points["mag"], points["snr"], points["isdiffpos"])]
    points["survey"] = survey_label
    return points


def combined_points(decam_row, boom_row) -> pd.DataFrame:
    parts = []
    if decam_row is not None and decam_row["n_points"] > 0:
        parts.append(decam_points(decam_row))
    if boom_row is not None:
        for prefix, label in (("ztf", "ZTF"), ("lsst", "LSST")):
            if isinstance(boom_row.get(f"{prefix}name"), str):
                parts.append(boom_points(boom_row, prefix, label))
    if not parts:
        return pd.DataFrame({"mjd": [], "filter": [], "mag": [], "magerr": [], "limiting_mag": [], "snr": [],
                             "detected": [], "survey": []}).astype(
            {"mjd": float, "mag": float, "magerr": float, "limiting_mag": float, "snr": float, "detected": bool})
    return pd.concat(parts, ignore_index=True)


def plot_lightcurve(points: pd.DataFrame, title: str, out_path: pathlib.Path) -> int:
    """Save the combined light curve. Returns how many points had nothing to draw."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    detections = points[points["detected"] & np.isfinite(points["mjd"])]
    limits = points[~points["detected"] & np.isfinite(points["mjd"]) & np.isfinite(points["limiting_mag"])]
    n_undrawable = len(points) - len(detections) - len(limits)

    for (survey, filter_name), group in limits.groupby(["survey", "filter"], dropna=False):
        color = FILTER_COLORS.get(filter_name, OTHER_FILTER_COLOR)
        ax.scatter(group["mjd"], group["limiting_mag"], marker=UPPER_LIMIT_MARKER, s=34,
                   facecolors="none", edgecolors=color, linewidths=1.0, alpha=UPPER_LIMIT_ALPHA, zorder=2)
    for (survey, filter_name), group in detections.groupby(["survey", "filter"], dropna=False):
        color = FILTER_COLORS.get(filter_name, OTHER_FILTER_COLOR)
        ax.errorbar(group["mjd"], group["mag"], yerr=group["magerr"], fmt=SURVEY_MARKERS[survey],
                    markersize=6, markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.6,
                    ecolor=color, elinewidth=0.8, capsize=0, linestyle="none", zorder=3)

    if len(detections) + len(limits):
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "no light-curve points", transform=ax.transAxes, ha="center", va="center", color="0.4")
    ax.set_xlabel("MJD", color="0.25")
    ax.set_ylabel("Magnitude", color="0.25")
    ax.grid(color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="0.35", labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # two legends, so a filter colour is never confused with a survey marker
    filters_present = [f for f in FILTER_COLORS if f in set(points["filter"])]
    if any(f not in FILTER_COLORS for f in set(points["filter"])):
        filters_present.append("other")
    filter_handles = [Line2D([], [], marker="s", linestyle="none", markersize=8, markeredgewidth=0,
                             markerfacecolor=FILTER_COLORS.get(f, OTHER_FILTER_COLOR),
                             label="N/A / other" if f == "other" else f) for f in filters_present]
    surveys_present = [s for s in SURVEY_MARKERS if s in set(points["survey"])]
    survey_handles = [Line2D([], [], marker=SURVEY_MARKERS[s], linestyle="none", markersize=7, markerfacecolor="white",
                             markeredgecolor="0.3", markeredgewidth=1.2, label="Rubin" if s == "LSST" else s)
                      for s in surveys_present]
    if len(limits):
        survey_handles.append(Line2D([], [], marker=UPPER_LIMIT_MARKER, linestyle="none", markersize=7,
                                     markerfacecolor="none", markeredgecolor="0.3", label="upper limit"))
    legend_style = dict(frameon=False, fontsize=9, title_fontsize=9, alignment="left")
    if filter_handles:
        filter_legend = ax.legend(handles=filter_handles, title="Filter", loc="upper left",
                                  bbox_to_anchor=(1.01, 1), **legend_style)
        ax.add_artist(filter_legend)
    if survey_handles:
        ax.legend(handles=survey_handles, title="Survey", loc="lower left", bbox_to_anchor=(1.01, 0), **legend_style)
    ax.set_title(title, loc="left", fontsize=11, color="0.15")
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return n_undrawable


# ---------------------------------------------------------------------------
# Cutouts
# ---------------------------------------------------------------------------

def plot_cutouts(decam_row, out_path: pathlib.Path) -> bool:
    """Save science / template / difference of the brightest point. False if the object has none."""
    if decam_row is None or decam_row["brightest_index"] < 0:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#3a3a3a")  # NaN pixels (e.g. off the chip edge) in grey, not black
    for ax, (column, label) in zip(axes, CUTOUT_NAMES.items()):
        stamp = np.asarray(decam_row[column], dtype="f4").reshape(CUTOUT_SIDE, CUTOUT_SIDE)
        finite = stamp[np.isfinite(stamp)]
        vmin, vmax = (np.percentile(finite, [1, 99.5]) if finite.size else (0, 1))
        ax.imshow(stamp, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
        ax.set_title(label, fontsize=10, color="0.2")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"DECam brightest point: MJD {decam_row['brightest_mjd']:.3f}, "
                 f"{decam_row['brightest_filter']} = {decam_row['brightest_mag']:.2f}",
                 x=0.01, ha="left", fontsize=10, color="0.25")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# scanner.html
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quick scan</title>
<style>
  :root {{ --bg: #f4f4f2; --card: #ffffff; --ink: #1d1d1b; --muted: #6b6b66; --line: #e2e2dd; --accent: #2d5f8a; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }}
  header {{ position: sticky; top: 0; z-index: 1; background: var(--bg); border-bottom: 1px solid var(--line);
            padding: 12px 16px; display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }}
  header h1 {{ font-size: 17px; margin: 0; }}
  header span {{ color: var(--muted); font-size: 13px; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 16px; }}
  section {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 16px;
             margin-bottom: 20px; scroll-margin-top: 64px; }}
  section.current {{ border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }}
  h2 {{ font-size: 17px; margin: 0 0 4px; font-family: ui-monospace, Menlo, monospace; }}
  .meta {{ color: var(--muted); font-size: 13px; margin-bottom: 12px; }}
  .meta b {{ color: var(--ink); font-weight: 600; }}
  img {{ display: block; max-width: 100%; height: auto; margin-top: 8px; }}
  .missing {{ color: var(--muted); font-style: italic; margin: 8px 0 0; }}
</style>
</head>
<body>
<header>
  <h1>Quick scan</h1>
  <span>{n_objects} objects &middot; {audience} &middot; {created}</span>
  <span>j / k: next / previous object</span>
</header>
<main>
{sections}
</main>
<script>
  const sections = [...document.querySelectorAll("section")];
  let current = -1;
  function go(i) {{
    if (!sections.length) return;
    current = Math.max(0, Math.min(sections.length - 1, i));
    sections.forEach((s, k) => s.classList.toggle("current", k === current));
    sections[current].scrollIntoView({{ behavior: "smooth", block: "start" }});
  }}
  document.addEventListener("keydown", (e) => {{
    if (e.target.closest("input, textarea")) return;
    if (e.key === "j") go(current + 1);
    if (e.key === "k") go(current - 1);
  }});
</script>
</body>
</html>
"""


def html_section(summary: dict) -> str:
    esc = html.escape
    meta = [f"RA <b>{summary['ra']:.5f}</b>, Dec <b>{summary['dec']:.5f}</b>" if summary["ra"] is not None else "no position",
            f"program <b>{esc(str(summary['program']))}</b>"]
    for label, key in (("ZTF", "ztfname"), ("Rubin", "lsstname")):
        if summary.get(key):
            meta.append(f"{label} <b>{esc(summary[key])}</b>")
    meta.append(f"{summary['n_points']} points")
    if summary["n_undrawable"]:
        meta.append(f"{summary['n_undrawable']} with no mag or limit (not drawn)")
    body = [f'<img src="{esc(summary["lightcurve_png"])}" alt="light curve of {esc(summary["objid"])}" loading="lazy">']
    if summary["cutouts_png"]:
        body.append(f'<img src="{esc(summary["cutouts_png"])}" alt="DECam cutouts of {esc(summary["objid"])}" loading="lazy">')
    else:
        body.append('<p class="missing">No DECam cutouts for this object.</p>')
    for note in summary["notes"]:
        body.append(f'<p class="missing">{esc(note)}</p>')
    return (f'<section id="{esc(summary["objid"])}">\n  <h2>{esc(summary["objid"])}</h2>\n'
            f'  <div class="meta">{" &middot; ".join(meta)}</div>\n  ' + "\n  ".join(body) + "\n</section>")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Light curves + DECam cutouts + scanner.html for a list of objects.")
    parser.add_argument("objid_list", type=pathlib.Path, help="Text file with one objid per line.")
    parser.add_argument("--collaborator", default=None,
                        help="Write to exports/collaborators/<name>/ instead of exports/me/.")
    args = parser.parse_args()

    objids = read_objid_list(args.objid_list)
    out_dir = output_dir_for(args.collaborator)
    lightcurve_dir, cutout_dir = out_dir / "plots" / "lightcurves", out_dir / "plots" / "cutouts"
    lightcurve_dir.mkdir(parents=True, exist_ok=True)
    cutout_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {out_dir}")

    master_rows = first_row_per_objid(read_rows_for(MASTER_FILE, "objid", objids), "objid", MASTER_FILE.name)
    decam_rows = first_row_per_objid(read_rows_for(DECAM_LIGHTCURVES_FILE, "objid", objids), "objid",
                                     DECAM_LIGHTCURVES_FILE.name)
    boom_columns = ["decam_objid"] + [f"{p}{suffix}" for p in ("ztf", "lsst") for suffix in ("name", "_sep_arcsec")] + \
                   [f"{p}_{field}" for p in ("ztf", "lsst") for field in BOOM_LIGHTCURVE_FIELDS]
    boom_rows = first_row_per_objid(read_rows_for(BOOM_LIGHTCURVES_FILE, "decam_objid", objids, boom_columns),
                                    "decam_objid", BOOM_LIGHTCURVES_FILE.name)

    sections, n_not_found = [], 0
    for objid in tqdm(objids, desc="Plotting", unit="object"):
        master_row, decam_row, boom_row = master_rows.get(objid), decam_rows.get(objid), boom_rows.get(objid)
        notes = []
        if master_row is None:
            notes.append("Not in the master file.")
        if decam_row is None:
            notes.append("No DECam light curve (not in decam_lightcurves.parquet).")
        elif isinstance(decam_row.get("read_error"), str):
            notes.append(f"DECam FITS file could not be read: {decam_row['read_error']}")
        if master_row is None and decam_row is None and boom_row is None:
            n_not_found += 1
            logger.warning(f"{objid}: not found in any file")

        ztfname = boom_row["ztfname"] if boom_row is not None and isinstance(boom_row["ztfname"], str) else None
        lsstname = boom_row["lsstname"] if boom_row is not None and isinstance(boom_row["lsstname"], str) else None
        title = objid + "".join(f"   {label}: {name}" for label, name in (("ZTF", ztfname), ("Rubin", lsstname)) if name)

        points = combined_points(decam_row, boom_row)
        lightcurve_png = lightcurve_dir / f"{objid}_lightcurve.png"
        n_undrawable = plot_lightcurve(points, title, lightcurve_png)
        cutouts_png = cutout_dir / f"{objid}_cutouts.png"
        has_cutouts = plot_cutouts(decam_row, cutouts_png)

        position_row = master_row if master_row is not None else decam_row
        sections.append(html_section({
            "objid": objid,
            "ra": float(position_row["ra"]) if position_row is not None and pd.notna(position_row["ra"]) else None,
            "dec": float(position_row["dec"]) if position_row is not None and pd.notna(position_row["dec"]) else None,
            "program": position_row["program"] if position_row is not None else "?",
            "ztfname": ztfname, "lsstname": lsstname,
            "n_points": len(points), "n_undrawable": n_undrawable,
            "lightcurve_png": lightcurve_png.relative_to(out_dir).as_posix(),
            "cutouts_png": cutouts_png.relative_to(out_dir).as_posix() if has_cutouts else None,
            "notes": notes,
        }))

    scanner = out_dir / "scanner.html"
    scanner.write_text(HTML_TEMPLATE.format(
        n_objects=len(objids),
        audience=html.escape(f"for {args.collaborator}" if args.collaborator else "for me"),
        created=datetime.now().strftime("%Y-%m-%d %H:%M"),
        sections="\n".join(sections),
    ))
    logger.info(f"Wrote {scanner} ({len(objids)} objects, {n_not_found} not found in any file)")


if __name__ == "__main__":
    main()
