"""
Sky coverage and cadence of the survey, from the master file (create_masterdb.py).

The sky is split into ~equal-area cells (--cell-deg on a side). For each cell, the MJD_OBS
of every candidate detected in it are pooled, and times closer than --visit-tol-min are
merged into one visit. A cell's visits are then the epochs at which it was observed and at
least one candidate in it was detected -- a lower limit on the true pointings, since
exposures with no detections in the cell leave no trace in the master file.

Per cell: n_objects, n_visits, n_nights, median_cadence_d (median gap between consecutive
visits), median_intranight_d (gaps within a night), median_internight_d (gaps between
consecutive nights), min_gap_d, frac_gaps_lt_1h, frac_gaps_lt_1d.

Outputs:
    data/sky_cadence.parquet                 the per-cell table
    plots/sky/skymap_visits.png              number of visits per cell
    plots/sky/skymap_cadence.png             median cadence per cell
    plots/sky/cadence_hist.png               all visit gaps, and median cadence per cell

Usage:
    python tidez/sky_cadence.py
    python tidez/sky_cadence.py --program DESIRT --cell-deg 0.25
    python tidez/sky_cadence.py --extent 30 60 -10 10        # zoom: ra_min ra_max dec_min dec_max
    python tidez/sky_cadence.py --ra 150.1 --dec -2.3        # print the cadence at one position
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, NullFormatter

from tidez.logger import get_logger
logger = get_logger("sky_cadence")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MASTER_FILE = DATA_DIR / "all_objects.parquet"
OUT_FILE = DATA_DIR / "sky_cadence.parquet"
PLOT_DIR = PROJECT_ROOT / "plots" / "sky"

CELL_DEG = 0.5
VISIT_TOL_MIN = 1.0
# Nights split at local noon at CTIO (UTC-4 -> 16:00 UTC = MJD fraction 0.667), so one
# night's exposures (UTC ~23:00 to ~10:00) share a night id.
NIGHT_SPLIT_FRAC = 0.667
HOUR, DAY = 1 / 24, 1.0
CADENCE_TICKS = [(1 / 1440, "1 min"), (10 / 1440, "10 min"), (HOUR, "1 h"), (6 * HOUR, "6 h"),
                 (DAY, "1 d"), (3 * DAY, "3 d"), (7 * DAY, "1 wk"), (30 * DAY, "30 d"),
                 (100 * DAY, "100 d"), (365 * DAY, "1 yr")]


def cadence_ticks_in(lo: float, hi: float) -> list:
    """CADENCE_TICKS within [lo, hi], with a little slack so a tick on the edge isn't lost to rounding."""
    return [(t, s) for t, s in CADENCE_TICKS if lo / 1.05 <= t <= hi * 1.05]


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def cell_index(ra, dec, cell_deg: float):
    """(dec_idx, ra_idx) of the ~equal-area cell holding each position: each dec band of height
    cell_deg is split into round(360 cos(dec) / cell_deg) ra cells."""
    ra, dec = np.asarray(ra, dtype=float) % 360, np.clip(np.asarray(dec, dtype=float), -90, 89.999999)
    dec_idx = np.floor((dec + 90) / cell_deg).astype(int)
    n_ra = n_ra_cells(dec_idx, cell_deg)
    ra_idx = np.minimum(np.floor(ra / 360 * n_ra).astype(int), n_ra - 1)
    return dec_idx, ra_idx


def n_ra_cells(dec_idx, cell_deg: float):
    dec_center = -90 + (np.asarray(dec_idx) + 0.5) * cell_deg
    return np.maximum(1, np.round(360 * np.cos(np.radians(dec_center)) / cell_deg)).astype(int)


def cell_center(dec_idx, ra_idx, cell_deg: float):
    n_ra = n_ra_cells(dec_idx, cell_deg)
    return (np.asarray(ra_idx) + 0.5) * 360 / n_ra, -90 + (np.asarray(dec_idx) + 0.5) * cell_deg


# ---------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------

def load_master(master: pathlib.Path, program: str | None) -> pd.DataFrame:
    if not master.exists():
        raise FileNotFoundError(f"{master} not found -- run create_masterdb.py first")
    df = pd.read_parquet(master, columns=["objid", "program", "ra", "dec", "mjds"])
    logger.info(f"Loaded {len(df)} objects from {master}")
    if program:
        df = df[df["program"] == program]
        logger.info(f"Kept {len(df)} objects from program {program}")
    n_before = len(df)
    df = df.dropna(subset=["ra", "dec"])
    df = df[df["mjds"].apply(len) > 0]
    if len(df) < n_before:
        logger.warning(f"Dropped {n_before - len(df)} objects with no ra/dec or no epochs")
    if df.empty:
        raise SystemExit(f"No objects with ra/dec and epochs in {master}" + (f" for {program}" if program else ""))
    return df


def compute_cell_cadence(df: pd.DataFrame, cell_deg: float, visit_tol_min: float) -> tuple:
    """(cells, gaps): one row per sky cell with its visit count and cadence stats (see module
    docstring), and every visit-to-visit gap in days."""
    dec_idx, ra_idx = cell_index(df["ra"], df["dec"], cell_deg)
    objects = pd.DataFrame({"dec_idx": dec_idx, "ra_idx": ra_idx, "mjds": df["mjds"].values})
    n_objects = objects.groupby(["dec_idx", "ra_idx"]).size().rename("n_objects")

    # one row per (cell, detection time), sorted so visits and gaps are vectorised diffs
    det = objects.explode("mjds").rename(columns={"mjds": "mjd"})
    det["mjd"] = det["mjd"].astype(float)
    det = det.sort_values(["dec_idx", "ra_idx", "mjd"], ignore_index=True)
    logger.info(f"Pooled {len(det)} detections into {len(n_objects)} cells of {cell_deg} deg")

    same_cell = (det["dec_idx"].diff() == 0) & (det["ra_idx"].diff() == 0)
    new_visit = ~same_cell | (det["mjd"].diff() > visit_tol_min / 1440)
    visits = det[new_visit].reset_index(drop=True)
    visits["night"] = np.floor(visits["mjd"] - NIGHT_SPLIT_FRAC).astype(int)
    logger.info(f"Merged detections into {len(visits)} visits (tolerance {visit_tol_min} min)")

    same_cell = (visits["dec_idx"].diff() == 0) & (visits["ra_idx"].diff() == 0)
    visits["gap"] = visits["mjd"].diff().where(same_cell)
    visits["intranight_gap"] = visits["gap"].where(visits["night"].diff() == 0)
    nights = visits.drop_duplicates(["dec_idx", "ra_idx", "night"]).copy()
    same_cell_n = (nights["dec_idx"].diff() == 0) & (nights["ra_idx"].diff() == 0)
    nights["internight_gap"] = nights["night"].diff().where(same_cell_n)

    keys = ["dec_idx", "ra_idx"]
    g = visits.groupby(keys)
    cells = pd.DataFrame({
        "n_visits": g.size(),
        "n_nights": g["night"].nunique(),
        "first_mjd": g["mjd"].min(),
        "last_mjd": g["mjd"].max(),
        "median_cadence_d": g["gap"].median(),
        "median_intranight_d": g["intranight_gap"].median(),
        "min_gap_d": g["gap"].min(),
        "frac_gaps_lt_1h": g["gap"].apply(lambda s: (s.dropna() < HOUR).mean()),
        "frac_gaps_lt_1d": g["gap"].apply(lambda s: (s.dropna() < DAY).mean()),
    }).join(nights.groupby(keys)["internight_gap"].median().rename("median_internight_d")).join(n_objects)
    cells = cells.reset_index()
    cells["ra"], cells["dec"] = cell_center(cells["dec_idx"], cells["ra_idx"], cell_deg)
    cells["cell_deg"] = cell_deg
    return cells, visits["gap"].dropna().to_numpy()


def cadence_at(cells: pd.DataFrame, ra: float, dec: float) -> pd.Series | None:
    """The cell row holding (ra, dec), or None if nothing was detected there."""
    dec_idx, ra_idx = cell_index([ra], [dec], cells["cell_deg"].iloc[0])
    hit = cells[(cells["dec_idx"] == dec_idx[0]) & (cells["ra_idx"] == ra_idx[0])]
    return None if hit.empty else hit.iloc[0]


def format_days(d: float) -> str:
    if pd.isna(d):
        return "n/a"
    if d < HOUR:
        return f"{d * 1440:.1f} min"
    if d < DAY:
        return f"{d * 24:.1f} h"
    return f"{d:.1f} d"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def sky_axes(extent):
    """Mollweide all-sky axes (RA increasing to the left), or plain RA/Dec axes for an extent."""
    fig = plt.figure(figsize=(11, 6) if extent is None else (9, 7))
    if extent is None:
        ax = fig.add_subplot(projection="mollweide")
        ax.set_xticks(np.radians([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]))
        ax.set_xticklabels([f"{v}°" for v in [330, 300, 270, 240, 210, 180, 150, 120, 90, 60, 30]],
                           color="0.4", fontsize=8)
    else:
        ax = fig.add_subplot()
        ra_min, ra_max, dec_min, dec_max = extent
        ax.set_xlim(ra_max, ra_min)  # RA increases to the left
        ax.set_ylim(dec_min, dec_max)
        ax.set_aspect(1 / np.cos(np.radians((dec_min + dec_max) / 2)))
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
    ax.grid(color="0.85", linewidth=0.5)
    ax.tick_params(colors="0.4", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("0.7")
    return fig, ax


def sky_xy(cells: pd.DataFrame, extent):
    """Mollweide wants radians in [-pi, pi]: RA 180 at the centre, growing leftward."""
    return -np.radians(cells["ra"] % 360 - 180), np.radians(cells["dec"])


def plot_skymap(cells: pd.DataFrame, column: str, label: str, title: str, out: pathlib.Path,
                extent, cmap: str, cadence_ticks: bool = False) -> None:
    """Cells coloured by `column` on a log scale. Cadence maps reverse the colormap, so the
    fastest-cadence cells are the darkest and are drawn on top."""
    cells = cells.dropna(subset=[column])
    cells = cells[cells[column] > 0].sort_values(column, ascending=not cadence_ticks)  # notable on top
    if extent is not None:
        ra_min, ra_max, dec_min, dec_max = extent
        cells = cells[cells["ra"].between(ra_min, ra_max) & cells["dec"].between(dec_min, dec_max)]
    if cells.empty:
        logger.warning(f"No cells with {column} to plot, skipping {out}")
        return

    fig, ax = sky_axes(extent)
    steps = np.linspace(1, 0.25, 256) if cadence_ticks else np.linspace(0.25, 1, 256)  # skip the near-white end
    cm = LinearSegmentedColormap.from_list(cmap + "_trunc", plt.get_cmap(cmap)(steps))
    vmin, vmax = cells[column].min(), cells[column].max()
    log = vmax / vmin >= 10  # a log scale over less than a decade only adds clutter
    norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)
    if extent is None:  # cells are ~1 px at full-sky scale, so markers read better than patches
        x, y = sky_xy(cells, extent)
        sc = ax.scatter(x, y, c=cells[column], s=3, marker="s", linewidths=0, cmap=cm, norm=norm)
    else:  # draw each cell as its true RA/Dec rectangle
        width = 360 / n_ra_cells(cells["dec_idx"], cells["cell_deg"].iloc[0])
        height = cells["cell_deg"].iloc[0]
        rects = [Rectangle((ra - w / 2, dec - height / 2), w, height)
                 for ra, dec, w in zip(cells["ra"], cells["dec"], width)]
        sc = PatchCollection(rects, cmap=cm, norm=norm, linewidths=0)
        sc.set_array(cells[column].to_numpy())
        ax.add_collection(sc)
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05, aspect=40)
    cbar.set_label(f"{label} (log)" if log else label, color="0.25")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors="0.4", labelsize=8)
    if cadence_ticks:
        ticks = cadence_ticks_in(vmin, vmax)
        cbar.ax.xaxis.set_minor_formatter(NullFormatter())
        if ticks:
            cbar.set_ticks([t for t, _ in ticks], labels=[s for _, s in ticks])
    else:
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        cbar.ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(title, color="0.15", fontsize=12, loc="left", pad=14)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Wrote {out}")


def plot_cadence_hist(cells: pd.DataFrame, gaps: np.ndarray, title: str, out: pathlib.Path) -> None:
    gaps = gaps[gaps > 0]
    medians = cells["median_cadence_d"].dropna()
    medians = medians[medians > 0]
    if gaps.size == 0:
        logger.warning(f"No visit gaps to plot, skipping {out}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, values, xlabel, ylabel in [
        (axes[0], gaps, "Gap between consecutive visits", "Number of gaps"),
        (axes[1], medians.to_numpy(), "Median cadence of a cell", "Number of cells"),
    ]:
        if values.size == 0:
            continue
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()) + 1e-9, 60)
        ax.hist(values, bins=bins, color="#3b6ea8", edgecolor="white", linewidth=0.5)
        ax.set_xscale("log")
        ticks = cadence_ticks_in(values.min(), values.max())
        ax.set_xticks([t for t, _ in ticks], labels=[s for _, s in ticks])
        ax.minorticks_off()
        for t in (HOUR, DAY):
            ax.axvline(t, color="0.55", linewidth=1, linestyle="--")
        ax.set_xlabel(xlabel, color="0.25")
        ax.set_ylabel(ylabel, color="0.25")
        ax.tick_params(colors="0.4", labelsize=8)
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_title(f"{100 * np.mean(gaps < HOUR):.0f}% of gaps < 1 h, {100 * np.mean(gaps < DAY):.0f}% < 1 d",
                      color="0.35", fontsize=9, loc="left")
    fig.suptitle(title, color="0.15", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def log_position(cells: pd.DataFrame, ra: float, dec: float) -> None:
    row = cadence_at(cells, ra, dec)
    if row is None:
        logger.info(f"No detections in the cell holding RA={ra}, Dec={dec}")
        return
    logger.info(
        f"Cell at RA={row['ra']:.3f}, Dec={row['dec']:.3f} ({row['cell_deg']} deg) holding RA={ra}, Dec={dec}: "
        f"{int(row['n_objects'])} objects, {int(row['n_visits'])} visits over {int(row['n_nights'])} nights "
        f"(MJD {row['first_mjd']:.2f}-{row['last_mjd']:.2f}); "
        f"median cadence {format_days(row['median_cadence_d'])}, "
        f"intra-night {format_days(row['median_intranight_d'])}, "
        f"inter-night {format_days(row['median_internight_d'])}, "
        f"min gap {format_days(row['min_gap_d'])}; "
        f"{100 * row['frac_gaps_lt_1h']:.0f}% of gaps < 1 h, {100 * row['frac_gaps_lt_1d']:.0f}% < 1 d"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sky coverage and cadence maps from the master file.")
    parser.add_argument("--master", type=pathlib.Path, default=MASTER_FILE)
    parser.add_argument("--out", type=pathlib.Path, default=OUT_FILE)
    parser.add_argument("--plot-dir", type=pathlib.Path, default=PLOT_DIR)
    parser.add_argument("--program", default=None, help="Only use this program (e.g. DESIRT).")
    parser.add_argument("--cell-deg", type=float, default=CELL_DEG, help="Sky cell size in degrees.")
    parser.add_argument("--visit-tol-min", type=float, default=VISIT_TOL_MIN,
                        help="Detections in a cell closer than this (minutes) are one visit.")
    parser.add_argument("--extent", type=float, nargs=4, metavar=("RA_MIN", "RA_MAX", "DEC_MIN", "DEC_MAX"),
                        help="Plot this RA/Dec box instead of the whole sky.")
    parser.add_argument("--ra", type=float, help="Print the cadence at this RA (needs --dec).")
    parser.add_argument("--dec", type=float, help="Print the cadence at this Dec (needs --ra).")
    args = parser.parse_args()
    if (args.ra is None) != (args.dec is None):
        parser.error("--ra and --dec go together")

    df = load_master(args.master, args.program)
    cells, gaps = compute_cell_cadence(df, args.cell_deg, args.visit_tol_min)
    logger.info(f"{len(cells)} cells; visits per cell: median {cells['n_visits'].median():.0f}, "
                f"max {cells['n_visits'].max()}; median cadence over cells: "
                f"{format_days(cells['median_cadence_d'].median())}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(args.out, index=False)
    logger.info(f"Wrote {len(cells)} cells to {args.out}")

    if args.ra is not None:
        log_position(cells, args.ra, args.dec)

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    what = args.program or "all programs"
    suffix = f"_{args.program}" if args.program else ""
    plot_skymap(cells, "n_visits", "Visits per cell",
                f"Visits per {args.cell_deg}° cell, {what}",
                args.plot_dir / f"skymap_visits{suffix}.png", args.extent, "Blues")
    plot_skymap(cells, "median_cadence_d", "Median time between visits, darker = faster",
                f"Median cadence per {args.cell_deg}° cell, {what}",
                args.plot_dir / f"skymap_cadence{suffix}.png", args.extent, "Purples", cadence_ticks=True)
    plot_cadence_hist(cells, gaps, f"Revisit gaps, {what}", args.plot_dir / f"cadence_hist{suffix}.png")


if __name__ == "__main__":
    main()
