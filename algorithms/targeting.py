"""
Choose *where* inside a UI element to click.

People don't click the exact geometric centre: clicks scatter around the
centre and very rarely land near the edges. We sample a Gaussian around the
centre and keep it inside an inner margin of the element.

(The mouse datasets don't record element boundaries, so this part is a
parametric model rather than learned. Tune `spread` / `margin` if needed.)
"""

import numpy as np


def pick_click_point(rect, rng, spread=0.17, margin=0.15, max_span=260, min_margin_px=2):
    """
    Return an (x, y) integer point inside rect = (left, top, right, bottom).

    spread:   std-dev as a fraction of the element size.
    margin:   fraction of each side that is never clicked.
    max_span: very wide elements (e.g. the address bar) are treated as at most
              this many pixels wide around their centre, like a person aiming
              somewhere near the middle rather than anywhere along 1,500 px.
    """
    left, top, right, bottom = rect
    w, h = max(right - left, 1), max(bottom - top, 1)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    span_w = min(w, max_span)

    x = cx + rng.normal(0, spread * span_w)
    y = cy + rng.normal(0, spread * h)

    mx = max(margin * w, min_margin_px)
    my = max(margin * h, min_margin_px)
    x = np.clip(x, left + mx, right - mx) if w > 2 * mx else cx
    y = np.clip(y, top + my, bottom - my) if h > 2 * my else cy
    return int(round(x)), int(round(y))


def random_point_in(rect, rng, inset=0.1):
    """Uniform random point inside rect, keeping `inset` fraction away from edges."""
    left, top, right, bottom = rect
    w, h = right - left, bottom - top
    return (int(rng.uniform(left + inset * w, right - inset * w)),
            int(rng.uniform(top + inset * h, bottom - inset * h)))


def primary_aim(rect, point, start, rng, along=0.012, across=0.006, undershoot=0.008):
    """
    Where the fast, primary part of an aimed movement actually lands.

    Aimed movements are a fast primary movement followed, when it misses, by a short
    corrective one (Meyer et al.'s optimised-submovement model, which underlies Fitts's
    law). The primary endpoint scatters in proportion to the distance moved - more
    along the movement direction than across it, with a slight undershoot.

    Small targets are therefore often missed and need a correction; large ones almost
    never. Returns (x, y), which may lie outside `rect`.
    """
    sx, sy = start
    tx, ty = point
    d = float(np.hypot(tx - sx, ty - sy))
    if d < 1:
        return point
    ux, uy = (tx - sx) / d, (ty - sy) / d
    a = rng.normal(-undershoot * d, along * d + 1.0)     # along the movement
    c = rng.normal(0.0, across * d + 1.0)                # sideways
    return int(round(tx + a * ux - c * uy)), int(round(ty + a * uy + c * ux))


def inside(point, rect, pad=0):
    """True if point lies inside rect shrunk by `pad` pixels on every side."""
    return rect[0] + pad <= point[0] <= rect[2] - pad and rect[1] + pad <= point[1] <= rect[3] - pad
