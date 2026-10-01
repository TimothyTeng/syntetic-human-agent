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

from controller import mouse, ui_elements

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


class HumanMouse:
    def __init__(self, model_path=DEFAULT_MODEL, stats_path=DEFAULT_STATS, use_model=True,
                 temperature=1.0, speed=1.0, seed=None, resample_hz=RESAMPLE_HZ):
        """
        model_path:  trained .npz from train.py (ignored if missing or use_model=False).
        stats_path:  click_stats.json from train.py (defaults used if missing).
        temperature: model sampling temperature (<1 calmer, >1 more varied).
        speed:       playback speed multiplier (1 = recorded human speed).
        seed:        random seed for reproducible runs (None = random).
        resample_hz: fill coarse model samples up to this polling rate during playback
                     (None = play the model's samples as generated).
        """
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature
        self.speed = speed
        self.resample_hz = resample_hz
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
        """Generate (but don't play) a trajectory [(x, y, dt), ...] from start to end."""
        traj = None
        # the model never saw moves under 15 px (filtered out in training): use the fallback
        if self.model and np.hypot(end[0] - start[0], end[1] - start[1]) >= MIN_MODEL_DIST:
            traj = self.model.generate(start, end, self.rng, temperature=self.temperature)
        if traj is None:  # no model, or model failed to reach the target
            traj = fallback.generate(start, end, self.rng)
        return traj

    def move_to(self, x, y):
        """Move from the current cursor position to (x, y) along a human-like path."""
        start = mouse.get_position()
        if np.hypot(x - start[0], y - start[1]) < 2:
            mouse.move_to(x, y)
            return
        player.play_trajectory(self.plan(start, (x, y)), speed=self.speed,
                               resample_hz=self.resample_hz, rng=self.rng)

    def move_to_rect(self, rect):
        """
        Move to a human-chosen point inside rect (left, top, right, bottom). Returns the point.

        The primary movement lands with distance-dependent scatter
        (targeting.primary_aim); if that is outside the element (common for small
        buttons, rare for big ones), the person notices and makes a short correction.
        """
        x, y = targeting.pick_click_point(rect, self.rng)
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

    def click_rect(self, rect, button="left"):
        """Move into rect (left, top, right, bottom) and click. Returns the clicked point."""
        point = self.move_to_rect(rect)
        self.press_release(button)
        return point

    def click_element(self, ctrl, button="left"):
        """Click a uiautomation control (from controller.ui_elements)."""
        return self.click_rect(ui_elements.element_rect(ctrl), button)

    def double_click_rect(self, rect):
        """Move into rect and double-click (second click follows within ~0.1-0.2 s)."""
        self.move_to_rect(rect)
        self.press_release()
        time.sleep(self.rng.uniform(0.06, 0.16))   # gap between the two clicks
        mouse.mouse_down()
        time.sleep(self.clicks.sample_hold(self.rng))
        mouse.mouse_up()
