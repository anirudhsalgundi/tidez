"""
Render the Legacy Survey cutout stored in the master H5 (see
crossmatch_legacy_survey.py) as a base64-encoded PNG with a crosshair +
circle marker at the target position, for use by the web scanner.
"""

import base64
import io

import numpy as np
from PIL import Image, ImageDraw

MARKER_COLOR = "#39ff14"
CROSSHAIR_LEN = 20
CROSSHAIR_GAP = 5
CIRCLE_RADIUS_ARCSEC = 3.0


def _render_png(jpg_bytes: bytes, pixscale: float) -> str:
    img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    # The cutout is always centered on the target RA/Dec (see
    # crossmatch_legacy_survey.py's cutout request), so the crosshair sits
    # at the image midpoint.
    cx, cy = img.width / 2, img.height / 2

    draw.line([(cx - CROSSHAIR_LEN, cy), (cx - CROSSHAIR_GAP, cy)], fill=MARKER_COLOR, width=2)
    draw.line([(cx + CROSSHAIR_GAP, cy), (cx + CROSSHAIR_LEN, cy)], fill=MARKER_COLOR, width=2)
    draw.line([(cx, cy - CROSSHAIR_LEN), (cx, cy - CROSSHAIR_GAP)], fill=MARKER_COLOR, width=2)
    draw.line([(cx, cy + CROSSHAIR_GAP), (cx, cy + CROSSHAIR_LEN)], fill=MARKER_COLOR, width=2)

    circle_radius_pix = CIRCLE_RADIUS_ARCSEC / pixscale
    draw.ellipse(
        [cx - circle_radius_pix, cy - circle_radius_pix, cx + circle_radius_pix, cy + circle_radius_pix],
        outline=MARKER_COLOR, width=2,
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def get_object_legacy_cutout(h5file, objid: str) -> dict | None:
    """Return {'image': b64png, 'layer': str, 'pixscale': float} for the
    object's stored Legacy Survey cutout, or None if it hasn't been fetched
    (see crossmatch_legacy_survey.py)."""
    group = h5file[objid].get("legacy_survey")
    if group is None:
        return None

    jpg_bytes = group["cutout_jpg"][:].tobytes()
    pixscale = float(group.attrs["pixscale"])
    layer = str(group.attrs["layer"])

    return {
        "image": _render_png(jpg_bytes, pixscale),
        "layer": layer,
        "pixscale": pixscale,
    }
