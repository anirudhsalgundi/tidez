"""
Interactive slide-deck style viewer for the master H5's lightcurves.

- Prev/Next buttons step through objects one at a time.
- Favorite button appends the current objid to favorite_sources.txt
  (skips duplicates; button label reflects current favorited state).
- Double-click a data point or a legend entry to dim everything else and
  highlight just that series (and its legend entry). Double-click empty
  axes space to reset.

Usage:
    python -m tidez.lightcurve_viewer
    python -m tidez.lightcurve_viewer --h5 data/tidez_master.h5
    python -m tidez.lightcurve_viewer --objids-file data/boom_crossmatches.txt
"""

import argparse
import pathlib

import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from tidez.logger import get_logger
from tidez.plot_lightcurve import (
    DEFAULT_H5,
    crossmatch_title_suffix,
    load_object_rows,
    render_lightcurve,
)

logger = get_logger("lightcurve_viewer")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_FAVORITES_FILE = DATA_DIR / "favorite_sources.txt"

DIM_ALPHA = 0.08


class LightcurveViewer:
    def __init__(self, h5_path: pathlib.Path, objids: list, favorites_file: pathlib.Path, show: bool = True):
        if not objids:
            raise ValueError("No objids to display.")

        self.h5_path = h5_path
        self.objids = objids
        self.favorites_file = favorites_file
        self.index = 0
        self.favorites = self._load_favorites()

        self.selected_label = None
        self.ax = None
        self.ax2 = None
        self.legend = None
        self.legend_labels = []
        self.artists_by_label = {}

        self.fig = plt.figure(figsize=(12, 6))
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self._draw_current()

        if show:
            plt.show()

    # -- favorites -----------------------------------------------------

    def _load_favorites(self):
        if self.favorites_file.exists():
            with open(self.favorites_file) as f:
                return {line.strip() for line in f if line.strip()}
        return set()

    def _on_favorite(self, event):
        objid = self.objids[self.index]
        if objid not in self.favorites:
            self.favorites.add(objid)
            self.favorites_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.favorites_file, "a") as f:
                f.write(objid + "\n")
            logger.info(f"Added to favorites: {objid} -> {self.favorites_file}")
        self._update_favorite_button_label()
        self.fig.canvas.draw_idle()

    def _update_favorite_button_label(self):
        objid = self.objids[self.index]
        self.btn_fav.label.set_text("★ Favorited" if objid in self.favorites else "☆ Favorite")

    # -- navigation ------------------------------------------------------

    def _on_prev(self, event):
        self.index = (self.index - 1) % len(self.objids)
        self._draw_current()

    def _on_next(self, event):
        self.index = (self.index + 1) % len(self.objids)
        self._draw_current()

    # -- drawing ---------------------------------------------------------

    def _build_widgets(self):
        ax_prev = self.fig.add_axes([0.35, 0.02, 0.1, 0.05])
        ax_next = self.fig.add_axes([0.46, 0.02, 0.1, 0.05])
        ax_fav = self.fig.add_axes([0.58, 0.02, 0.14, 0.05])

        self.btn_prev = Button(ax_prev, "< Prev")
        self.btn_next = Button(ax_next, "Next >")
        self.btn_fav = Button(ax_fav, "☆ Favorite")

        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_fav.on_clicked(self._on_favorite)

    def _draw_current(self):
        objid = self.objids[self.index]
        self.fig.clf()

        self.selected_label = None
        self.legend = None
        self.legend_labels = []
        self.artists_by_label = {}

        ax = self.fig.add_axes([0.08, 0.22, 0.60, 0.62])

        with h5py.File(self.h5_path, "r") as h5file:
            rows = load_object_rows(h5file, objid)
            title_suffix = crossmatch_title_suffix(h5file, objid)

        title_extra = f"  [{self.index + 1}/{len(self.objids)}]"
        ax2, legend, artists_by_label = render_lightcurve(
            ax, rows, objid, title_suffix=title_suffix, title_extra=title_extra
        )

        self.ax = ax
        self.ax2 = ax2
        self.legend = legend
        self.artists_by_label = artists_by_label
        if legend is not None:
            self.legend_labels = [t.get_text() for t in legend.get_texts()]
            for lh in legend.legend_handles:
                lh.set_picker(5)

        self._build_widgets()
        self._update_favorite_button_label()
        self.fig.canvas.draw_idle()

    # -- highlight on double-click ---------------------------------------

    @staticmethod
    def _base_alpha(label):
        return 0.35 if "upper limit" in label else 1.0

    def _set_artist_alpha(self, artist, alpha):
        if hasattr(artist, "lines"):  # ErrorbarContainer
            data_line, caplines, barlinecols = artist.lines
            if data_line is not None:
                data_line.set_alpha(alpha)
            for cap in caplines:
                cap.set_alpha(alpha)
            for coll in barlinecols:
                coll.set_alpha(alpha)
        else:  # PathCollection (scatter)
            artist.set_alpha(alpha)

    def _set_highlight(self, label):
        self.selected_label = label

        for lbl, artists in self.artists_by_label.items():
            highlighted = label is None or lbl == label
            alpha = self._base_alpha(lbl) if highlighted else DIM_ALPHA
            for artist in artists:
                self._set_artist_alpha(artist, alpha)

        if self.legend is not None:
            for text, lbl in zip(self.legend.get_texts(), self.legend_labels):
                highlighted = label is None or lbl == label
                text.set_alpha(1.0 if highlighted else 0.3)
                text.set_fontweight("bold" if (label is not None and lbl == label) else "normal")
            for lh, lbl in zip(self.legend.legend_handles, self.legend_labels):
                highlighted = label is None or lbl == label
                lh.set_alpha(1.0 if highlighted else 0.15)

        self.fig.canvas.draw_idle()

    def _hit_test(self, event):
        """Return the label of whatever data/legend artist was double-clicked, or None."""
        if self.legend is not None:
            for lh, lbl in zip(self.legend.legend_handles, self.legend_labels):
                contains, _ = lh.contains(event)
                if contains:
                    return lbl

        for lbl, artists in self.artists_by_label.items():
            for artist in artists:
                target = artist.lines[0] if hasattr(artist, "lines") else artist
                if target is None:
                    continue
                contains, _ = target.contains(event)
                if contains:
                    return lbl

        return None

    def _on_click(self, event):
        if not event.dblclick:
            return
        if self.ax is None:
            return

        label = self._hit_test(event)
        if label is not None:
            self._set_highlight(label)
        elif self.selected_label is not None:
            self._set_highlight(None)


def load_objids(h5_path: pathlib.Path, objids_file: pathlib.Path = None):
    if objids_file is not None:
        with open(objids_file) as f:
            return [line.strip() for line in f if line.strip()]

    with h5py.File(h5_path, "r") as h5file:
        return sorted(k for k in h5file.keys() if k != "catalog")


def main():
    parser = argparse.ArgumentParser(description="Interactive slide-deck viewer for tidez lightcurves.")
    parser.add_argument("--h5", type=pathlib.Path, default=DEFAULT_H5,
                         help="Master H5 file to read from.")
    parser.add_argument("--objids-file", type=pathlib.Path, default=None,
                         help="Text file of objids to restrict the deck to "
                              "(e.g. data/boom_crossmatches.txt). Defaults to all objects in the H5.")
    parser.add_argument("--favorites-file", type=pathlib.Path, default=DEFAULT_FAVORITES_FILE,
                         help="Text file to append favorited objids to.")
    args = parser.parse_args()

    objids = load_objids(args.h5, args.objids_file)
    logger.info(f"Loaded {len(objids)} objects to browse.")
    LightcurveViewer(args.h5, objids, args.favorites_file)


if __name__ == "__main__":
    main()
