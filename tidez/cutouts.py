"""
Render DECam cutout stamps (template/science/diff) stored in the master H5
as base64-encoded PNGs, for use by the web scanner.
"""

import base64
import io

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from matplotlib.figure import Figure
from matplotlib.patches import Circle

_ZSCALE = ZScaleInterval()

CUTOUT_KEYS = ("template", "science", "diff")

# Stamps are cut out centered on the source (see build_h5.py/STAMP_SIDE),
# so the crosshair always sits at the array midpoint.
MARKER_RADIUS = 8
MARKER_COLOR = "#39ff14"

# Diff-image contrast: real transient flux is small relative to occasional
# bad-subtraction outlier pixels, so scaling off the raw min/max (or even a
# high percentile of |value|) washes everything else out. Sigma-clipped
# stats reject those outliers and scale to the actual background noise.
DIFF_SIGMA_CLIP = 3.0
DIFF_VMAX_SIGMA = 4.0


def _render_png(array: np.ndarray, diverging: bool) -> str:
    arr = np.nan_to_num(np.asarray(array, dtype="f4"), nan=0.0)

    if diverging:
        _, _, std = sigma_clipped_stats(arr, sigma=DIFF_SIGMA_CLIP, maxiters=5)
        vmax = float(DIFF_VMAX_SIGMA * std) or 1.0
        vmin = -vmax
    else:
        vmin, vmax = _ZSCALE.get_limits(arr)

    fig = Figure(figsize=(2.4, 2.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr, cmap="inferno", vmin=vmin, vmax=vmax, origin="lower")

    center = ((arr.shape[1] - 1) / 2, (arr.shape[0] - 1) / 2)
    ax.add_patch(Circle(center, radius=MARKER_RADIUS, edgecolor=MARKER_COLOR,
                         facecolor="none", linewidth=1.3))

    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def get_object_cutouts(h5file, objid: str, epoch_idx: int) -> dict:
    """Return {'template': b64png, 'science': b64png, 'diff': b64png} for one epoch."""
    cutout_group = h5file[objid]["cutouts"]
    n_epochs = cutout_group["template"].shape[0]
    if not (0 <= epoch_idx < n_epochs):
        raise IndexError(f"epoch_idx {epoch_idx} out of range [0, {n_epochs}) for {objid}")

    return {
        key: _render_png(cutout_group[key][epoch_idx], diverging=(key == "diff"))
        for key in CUTOUT_KEYS
    }
