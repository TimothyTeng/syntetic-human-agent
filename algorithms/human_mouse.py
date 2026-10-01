"""
HumanMouse: the high-level mouse API used by tasks / main.py.

Wraps
  * the trained LSTM-MDN trajectory model (algorithms/mouse_model/sampler.py)
    or the fallback generator if no model file exists,
  * click timing learned from data (algorithms/mouse_model/click_stats.py),
  * click-point choice inside elements (algorithms/targeting.py),
  * accurate playback (algorithms/player.py),
and executes everything through controller.mouse.

Example:
    hm = HumanMouse()
    hm.click_rect(browser.get_address_bar_rect())
"""

import os
import time

import numpy as np

from controller import keyboard, mouse, screen, ui_elements

from . import player, targeting
from .mouse_model import fallback
from .mouse_model.click_stats import ClickStats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "mouse_mdn.npz")
DEFAULT_STATS = os.path.join(PROJECT_ROOT, "models", "click_stats.json")

# Playback polling rate. 100 Hz with the whole-pixel cap in player.resample() brings
# the model's mean sample spacing (~17 ms) to ~11 ms, close to real strokes (~12 ms).
RESAMPLE_HZ = 100
MIN_MODEL_DIST = 15     # px; shorter moves (corrections) use the fallback generator
LAND_TOLERANCE = 1      # px; a movement ending further than this from its target is corrected
MAX_CORRECTIONS = 2     # corrective movements per move before giving up

# Limits for an aimed movement, in units of the start->target distance D (plus a few px)
MAX_SIDEWAYS = 0.4      # sideways bulge
MAX_OVERSHOOT = 0.2     # past the target
MAX_BACKWARDS = 0.1     # behind the start
SLACK_PX = 20           # extra allowance so short moves aren't held to a few pixels
MAX_STEP_SPEED = 10000  # px/s between consecutive samples (dt taken as >= 8 ms); faster = a jump
MAX_TRIES = 15          # model samples before falling back


def aimed(traj, start, end, monitors=None):
    """
    True if trajectory [(x, y, dt), ...] looks like an aimed movement: it stays within
    MAX_SIDEWAYS * D of the start-target line, overshoots by at most MAX_OVERSHOOT * D,
    doesn't go back behind the start, never jumps (no step faster than MAX_STEP_SPEED)
    and (with `monitors`) never leaves the monitors.
    """
    pts = np.array([(x, y) for x, y, _ in traj], float)
    dts = np.clip([dt for *_, dt in traj], 0.008, None)
    steps = np.hypot(*np.diff(np.vstack([start, pts]), axis=0).T)
    if (steps / dts).max() > MAX_STEP_SPEED:
        return False
    v = np.subtract(end, start, dtype=float)
    D = float(np.hypot(*v))
    if D < 1:
        return True
    u = v / D
    rel = pts - np.asarray(start, float)
    along = rel @ u                                    # distance travelled towards the target
    side = np.abs(rel @ np.array([-u[1], u[0]]))      # distance from the start-target line
    if side.max() > MAX_SIDEWAYS * D + SLACK_PX:
        return False
    if along.max() > D * (1 + MAX_OVERSHOOT) + SLACK_PX or along.min() < -(MAX_BACKWARDS * D + SLACK_PX):
        return False
    if monitors and not all(screen.on_screen(x, y, monitors) for x, y in pts.round()):
        return False
    return True


class HumanMouse:
    def __init__(self, model_path=DEFAULT_MODEL, stats_path=DEFAULT_STATS, use_model=True,
                 temperature=1.0, speed=1.0, seed=None, resample_hz=RESAMPLE_HZ, click_spread=0.17):
        """
        model_path:  trained .npz from train.py (ignored if missing or use_model=False).
        stats_path:  click_stats.json from train.py (defaults used if missing).
        temperature: model sampling temperature (<1 calmer, >1 more varied).
        speed:       playback speed multiplier (1 = recorded human speed).
        seed:        random seed for reproducible runs (None = random).
        resample_hz: fill coarse model samples up to this polling rate during playback
                     (None = play the model's samples as generated).
        click_spread: scatter of click points around an element's centre, as a fraction
                     of its size (0.17 = typical; smaller = more precise clicks).
        """
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature
        self.speed = speed
        self.resample_hz = resample_hz
        self.click_spread = click_spread
        self.corrections = 0          # corrective movements made so far (diagnostics)
        self.rejected_plans = 0       # moves where no model sample was acceptable (fallback used)
        self.model = None
        if use_model and os.path.exists(model_path):
            from .mouse_model.sampler import MouseModel  # numpy only
            self.model = MouseModel.load(model_path)
        self.clicks = ClickStats.load(stats_path) if os.path.exists(stats_path) else ClickStats.default()

    @property
    def backend(self):
        """'lstm-mdn' if the trained model is loaded, else 'fallback'."""
        return "lstm-mdn" if self.model else "fallback"

    # --- Movement ------------------------------------------------------------------

    def plan(self, start, end):
        """
        Generate (but don't play) a trajectory [(x, y, dt), ...] from start to end.

        The model learned from strokes that sometimes wander far before the click
        (~1 in 3 overshoots or swings out by about the whole distance), which on screen
        reaches the edges, and occasionally jumps hundreds of pixels in one sample. Such
        strokes - and any leaving the monitors - are rejected and resampled (see
        aimed()); after MAX_TRIES the bounded fallback is used.
        """
        # the model never saw moves under 15 px (filtered out in training): use the fallback
        if self.model and np.hypot(end[0] - start[0], end[1] - start[1]) >= MIN_MODEL_DIST:
            monitors = screen.monitor_rects()
            for _ in range(MAX_TRIES):
                traj = self.model.generate(start, end, self.rng, temperature=self.temperature, retries=1)
                if traj is not None and aimed(traj, start, end, monitors):
                    return traj
            self.rejected_plans += 1
        return fallback.generate(start, end, self.rng)

    def move_to(self, x, y):
        """
        Move from the current cursor position to (x, y) along a human-like path.

        Closed loop, like a person watching the pointer: if the cursor doesn't end on
        the target (something else moved it, or a position was clamped), a short
        corrective movement follows after a brief reaction time. Returns True if the
        cursor ended within LAND_TOLERANCE px of (x, y).
        """
        x, y = int(round(x)), int(round(y))
        monitors = screen.monitor_rects()
        if not screen.on_screen(x, y, monitors, margin=2):   # e.g. a near-edge aim error or drift
            x, y = screen.nearest_on_screen(x, y, monitors, margin=2)
        start = mouse.get_position()
        if np.hypot(x - start[0], y - start[1]) < 2:
            mouse.move_to(x, y)
        else:
            player.play_trajectory(self.plan(start, (x, y)), speed=self.speed,
                                   resample_hz=self.resample_hz, rng=self.rng)
        for _ in range(MAX_CORRECTIONS):
            cx, cy = mouse.get_position()
            if abs(cx - x) <= LAND_TOLERANCE and abs(cy - y) <= LAND_TOLERANCE:
                return True
            self.corrections += 1
            time.sleep(float(0.1 * np.exp(self.rng.normal(0, 0.3))))      # notice, re-aim
            player.play_trajectory(fallback.generate((cx, cy), (x, y), self.rng), speed=self.speed)
        cx, cy = mouse.get_position()
        return abs(cx - x) <= LAND_TOLERANCE and abs(cy - y) <= LAND_TOLERANCE

    def move_to_rect(self, rect):
        """
        Move to a human-chosen point inside rect (left, top, right, bottom). Returns the point.

        The primary movement lands with distance-dependent scatter
        (targeting.primary_aim); if that is outside the element (common for small
        buttons, rare for big ones), the person notices and makes a short correction.
        """
        x, y = targeting.pick_click_point(rect, self.rng, spread=self.click_spread)
        start = mouse.get_position()
        landing = targeting.primary_aim(rect, (x, y), start, self.rng)
        if targeting.inside(landing, rect, pad=1):
            self.move_to(x, y)
            return x, y
        self.move_to(*landing)
        time.sleep(float(0.11 * np.exp(self.rng.normal(0, 0.35))))   # see the miss, re-aim
        self.move_to(x, y)
        return x, y

    def drift(self, max_dist=80, bounds=None):
        """
        Small idle movement to a nearby point (e.g. while reading).
        bounds: optional rect the point must stay inside.
        """
        x0, y0 = mouse.get_position()
        ang = self.rng.uniform(0, 2 * np.pi)
        dist = self.rng.uniform(15, max_dist)
        x, y = x0 + dist * np.cos(ang), y0 + dist * np.sin(ang)
        if bounds:
            x = float(np.clip(x, bounds[0] + 5, bounds[2] - 5))
            y = float(np.clip(y, bounds[1] + 5, bounds[3] - 5))
        self.move_to(int(x), int(y))

    # --- Clicking --------------------------------------------------------------------

    def press_release(self, button="left"):
        """Click at the current position with learned dwell + hold timing."""
        time.sleep(self.clicks.sample_dwell(self.rng))
        mouse.mouse_down(button)
        time.sleep(self.clicks.sample_hold(self.rng))
        mouse.mouse_up(button)

    def click_at(self, x, y, button="left"):
        """Move to (x, y) and click."""
        self.move_to(x, y)
        self.press_release(button)

    def click_rect(self, rect, button="left", refresh=None, modifiers=()):
        """
        Move into rect (left, top, right, bottom) and click. Returns the clicked point.

        refresh: optional callable returning the element's *current* rect (or None).
                 It is checked again just before pressing; if the element moved while
                 the cursor travelled (pages shift while loading), the person re-aims
                 into where it is now instead of clicking the old position.
        modifiers: keys held during the click, e.g. ('ctrl',) to open a link in a new tab.
        """
        point = self.move_to_rect(rect)
        if refresh is not None:
            now = refresh()
            if now and not targeting.inside(point, now, pad=1):
                self.corrections += 1
                time.sleep(float(0.12 * np.exp(self.rng.normal(0, 0.3))))
                point = targeting.pick_click_point(now, self.rng, spread=self.click_spread)
                self.move_to(*point)
        if modifiers:
            self.press_release_with(modifiers, button)
        else:
            self.press_release(button)
        return point

    def press_release_with(self, modifiers, button="left"):
        """Click while holding modifier keys: they go down a moment before the click and
        come up a moment after it. Never leaves a modifier held."""
        held = []
        try:
            for i, key in enumerate(modifiers):
                if i:
                    time.sleep(float(0.05 * np.exp(self.rng.normal(0, 0.5))))
                keyboard.key_down(key)
                held.append(key)
            time.sleep(float(0.15 * np.exp(self.rng.normal(0, 0.4))))     # modifier leads the click
            mouse.mouse_down(button)
            time.sleep(self.clicks.sample_hold(self.rng))
            mouse.mouse_up(button)
            time.sleep(float(0.07 * np.exp(self.rng.normal(0, 0.5))))     # ... and is let go after it
        finally:
            for key in reversed(held):
                keyboard.key_up(key)

    def click_element(self, ctrl, button="left", modifiers=()):
        """Click a uiautomation control (from controller.ui_elements), re-checking its
        position just before the click."""
        def current():
            try:
                return ui_elements.element_rect(ctrl) if ui_elements.is_visible(ctrl) else None
            except Exception:
                return None
        return self.click_rect(ui_elements.element_rect(ctrl), button, refresh=current, modifiers=modifiers)

    def double_click_rect(self, rect):
        """Move into rect and double-click (second click follows within ~0.1-0.2 s)."""
        self.move_to_rect(rect)
        self.press_release()
        time.sleep(self.rng.uniform(0.06, 0.16))   # gap between the two clicks
        mouse.mouse_down()
        time.sleep(self.clicks.sample_hold(self.rng))
        mouse.mouse_up()
