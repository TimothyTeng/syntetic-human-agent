"""
Cut raw sessions into "strokes": one point-and-click movement each.

A stroke is the run of Move events that ends in a left-button Press. The
cursor's resting position before the run is used as the starting point.
Runs are broken wherever the mouse was still for more than `gap` seconds.
The final point is where the click happened.

Also provides the canonical-frame transform used by the model:
    translate start -> (0,0), rotate so the target lies on +x, scale so the
    target is at (1,0). The pixel distance D is kept separately.
This makes one model work for every direction, distance and screen size.
"""

from dataclasses import dataclass

import numpy as np

from .dataset import LEFT, MOVE, PRESS, RELEASE


@dataclass
class Stroke:
    """One point-and-click movement."""
    points: np.ndarray   # (n, 2) screen pixels; points[0] = start, points[-1] = click position
    times: np.ndarray    # (n,) seconds, times[0] = 0
    user: str
    dwell: float         # seconds between the last movement and the button press
    hold: float          # seconds the button was held (nan if release not found)

    @property
    def distance(self):
        return float(np.linalg.norm(self.points[-1] - self.points[0]))

    @property
    def duration(self):
        return float(self.times[-1])


def segment_strokes(session, gap=0.5, min_dist=15.0, min_dur=0.08, max_dur=4.0,
                    min_points=5, max_points=200, window_filter=None, min_dt=0.0005):
    """
    Extract point-and-click strokes from one Session.

    gap:           pause (s) that separates two movements.
    min_dist:      ignore strokes shorter than this many pixels (micro-adjustments).
    min_dur/max_dur: duration limits in seconds.
    min_points/max_points: limits on number of samples in the stroke.
    window_filter: optional text; keep only clicks whose foreground window name
                   contains it (BMDD only), e.g. 'chrome'.
    min_dt:        events closer than this in time are merged (duplicate logs).

    Returns a list of Stroke.
    """
    t, x, y, st, bt = session.t, session.x, session.y, session.state, session.button
    n = len(t)
    strokes = []
    presses = np.flatnonzero((st == PRESS) & (bt == LEFT))

    for p in presses:
        if window_filter and session.window is not None:
            if window_filter.lower() not in str(session.window[p]).lower():
                continue

        # Walk backwards from the press over consecutive Move events
        k = p - 1
        chain = []
        while k >= 0 and st[k] == MOVE:
            if chain and (t[chain[-1]] - t[k]) > gap:
                break
            chain.append(k)
            k -= 1
        if not chain:
            continue
        chain.reverse()                       # chronological order
        first = chain[0]
        if t[p] - t[chain[-1]] > gap:          # long pause before clicking -> not one motion
            continue

        # Resting point before the run (event just before the chain), if any
        dts = np.diff(t[chain]) if len(chain) > 1 else np.array([0.016])
        typical_dt = float(np.median(dts[dts > 0])) if np.any(dts > 0) else 0.016
        if first - 1 >= 0:
            rest_xy = (x[first - 1], y[first - 1])
            rest_t = t[first] - min(t[first] - t[first - 1], typical_dt)
        else:
            rest_xy = (x[first], y[first])
            rest_t = t[first] - typical_dt

        pts = [rest_xy] + [(x[i], y[i]) for i in chain]
        tms = [rest_t] + [t[i] for i in chain]
        if (x[p], y[p]) != pts[-1]:          # press logged at a slightly different spot
            pts.append((x[p], y[p]))
            tms.append(tms[-1] + typical_dt)

        pts = np.asarray(pts, dtype=float)
        tms = np.asarray(tms, dtype=float)
        pts, tms = _merge_close_events(pts, tms, min_dt)

        # Click timing: dwell before press, hold until matching release
        dwell = float(t[p] - t[chain[-1]])
        hold = np.nan
        for r in range(p + 1, min(p + 50, n)):
            if st[r] == RELEASE and bt[r] == LEFT:
                if t[r] - t[p] < 2.0:
                    hold = float(t[r] - t[p])
                break

        stroke = Stroke(points=pts, times=tms - tms[0], user=session.user, dwell=dwell, hold=hold)
        if (len(pts) >= min_points and len(pts) <= max_points
                and stroke.distance >= min_dist and min_dur <= stroke.duration <= max_dur):
            strokes.append(stroke)
    return strokes


def _merge_close_events(pts, tms, min_dt):
    """Drop events that are (near-)duplicates in time, keeping the later one."""
    keep = np.ones(len(tms), dtype=bool)
    for i in range(1, len(tms)):
        if tms[i] - tms[i - 1] < min_dt:
            keep[i - 1] = False
    keep[0] = True  # always keep the start point
    return pts[keep], tms[keep]


# --- Canonical frame -------------------------------------------------------------

def rotation(angle):
    """2x2 rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def to_canonical(points):
    """
    Map screen points into the canonical frame.
    Returns (canon_points, D, angle) where canon_points[0] = (0,0), canon_points[-1] = (1,0).
    """
    start, end = points[0], points[-1]
    v = end - start
    D = float(np.hypot(*v))
    angle = float(np.arctan2(v[1], v[0]))
    canon = (points - start) @ rotation(-angle).T / D
    return canon, D, angle


def from_canonical(canon, start, D, angle):
    """Inverse of to_canonical: canonical points -> screen points."""
    return np.asarray(start, dtype=float) + D * (canon @ rotation(angle).T)
