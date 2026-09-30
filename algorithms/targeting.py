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
