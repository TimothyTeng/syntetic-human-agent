"""
Behaviour layer: one consistent simulated person.

    human = Human(seed=42)                 # samples a Persona, loads the mouse model
    human.open_app("word")
    human.click_field(rect); human.type("Quarterly summary")
    human.go_back()

Persona   per-person traits sampled once per run (typing speed, mouse speed,
          shortcut-vs-mouse preference, how long they think, how fast they read),
          so everything one run does fits the same "person".
Human     the facade tasks call. On top of HumanMouse / human_typing it adds
            * thinking time per step, after the Keystroke-Level Model (KLM):
              M = mental preparation (~1.35 s mean), shorter confirm / glance pauses
            * homing time (KLM H, ~0.4 s) whenever the hand moves between mouse and
              keyboard, inserted automatically from the device each action uses
            * visual-search time when choosing 1 of N items (Hick-Hyman:
              a + b*log2(N+1)), with people favouring the top of a list
            * method variety: Back button or Alt+Left, clicking the address bar or
              Ctrl+L, clicking Start or pressing Win - chosen per action from the
              persona's shortcut preference, so no action is always done one way
            * human-timed glue keys (Enter, Ctrl+L...) and typing of URLs, app
              names and paths through the learned typing model
            * fatigue and tempo drift (Fatigue): fresh at the start - a bit faster and
              more accurate than the persona's average - then slowly slower, sloppier
              and more hesitant as the session goes on; idle breaks partly recover.
              On top, the tempo wanders slowly up and down.
"""

import ctypes
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import uiautomation as auto

from controller import browser, config, mouse, office, screen, ui_elements

from . import human_typing
from .human_mouse import HumanMouse
from .mouse_model import scroll_stats


def _lognormal(rng, median, sigma):
    return float(median * np.exp(rng.normal(0, sigma)))


@dataclass
class Persona:
    wpm: float = 55.0               # typing speed (words per minute)
    mouse_speed: float = 1.0        # playback speed of mouse movements (1 = recorded human speed)
    mouse_temperature: float = 1.0  # variety of mouse paths
    shortcut_pref: float = 0.5      # 0 = reaches for the mouse, 1 = keyboard shortcuts
    think_scale: float = 1.0        # multiplies every thinking pause
    error_scale: float = 1.0        # multiplies the typo rate
    pause_scale: float = 0.7        # composition pauses (1.0 = timed-essay writers)
    reading_wpm: float = 230.0      # reading speed
    click_spread: float = 0.17      # click scatter around an element's centre (fraction of its size)
    new_tab_pref: float = 0.3       # how often links are opened in a new tab (Ctrl+click)
    fatigue_tau_min: float = 20.0   # minutes of work until ~63% of the way from fresh to tired
    maximize_pref: float = 0.7      # how often a newly opened window gets maximised
    scroll_profile: str = ""        # whose scroll-wheel style (a Balabit user) this person has

    @classmethod
    def sample(cls, rng, **overrides):
        """A random but plausible office worker. Keyword overrides (non-None) win."""
        p = cls(
            wpm=float(np.clip(rng.normal(55, 12), 30, 95)),
            mouse_speed=float(np.clip(np.exp(rng.normal(0, 0.12)), 0.75, 1.35)),
            shortcut_pref=float(rng.beta(2.0, 2.0)),
            think_scale=float(np.clip(np.exp(rng.normal(0, 0.25)), 0.6, 1.7)),
            error_scale=float(np.clip(np.exp(rng.normal(0, 0.3)), 0.5, 2.0)),
            reading_wpm=float(np.clip(rng.normal(230, 40), 150, 350)),
            new_tab_pref=float(rng.beta(1.5, 3.0)),
            fatigue_tau_min=float(np.clip(20 * np.exp(rng.normal(0, 0.3)), 10, 40)),
            maximize_pref=float(rng.beta(4.0, 1.7)),
        )
        styles = scroll_stats.load_default().profiles()
        if styles:
            p.scroll_profile = styles[int(rng.integers(len(styles)))]
        for k, v in overrides.items():
            if v is not None:
                setattr(p, k, v if isinstance(getattr(p, k), str) else float(v))
        return p

    def describe(self):
        return ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in asdict(self).items())


# (median s, log-normal sigma) of the pause before an action, by kind
THINK = {
    "mental": (1.2, 0.5),    # KLM M: deciding what to do next (mean ~1.35 s)
    "scan": (0.8, 0.5),      # looking over a new screen before acting
    "confirm": (0.4, 0.4),   # checking what was typed / selected before Enter
    "glance": (0.25, 0.4),   # a quick look
}
HOMING = (0.36, 0.3)         # KLM H: hand between mouse and keyboard (mean ~0.4 s)


@dataclass
class FatigueParams:
    # multipliers on the persona's averages, fresh -> tired
    speed_fresh: float = 1.08     # typing and mouse speed
    speed_tired: float = 0.88
    errors_fresh: float = 0.75    # typo rate, slips, click scatter
    errors_tired: float = 1.40
    think_fresh: float = 0.90     # thinking / searching pauses
    think_tired: float = 1.30
    wander_sd: float = 0.05       # slow random tempo drift (log speed), on top of fatigue
    wander_tau_s: float = 90.0    # ... how quickly that drift changes
    recovery: float = 1.5         # one second of rest undoes this many seconds of work


class Fatigue:
    """
    How fresh the person is. level() goes from 0 (fresh) towards 1 (tired) with the
    time worked: 1 - exp(-t / tau). factors() turns that into multipliers for speed,
    errors and thinking time, plus a slow random wander of the tempo (an
    Ornstein-Uhlenbeck process), so the pace drifts even within a few minutes.
    rest(seconds) - e.g. an idle break - takes work time off again.
    """

    def __init__(self, rng, tau_min=20.0, params=FatigueParams(), enabled=True):
        self.rng, self.tau, self.p, self.enabled = rng, tau_min * 60.0, params, enabled
        self.t0 = self.last = time.monotonic()
        self.rested = 0.0
        self.z = 0.0

    def level(self):
        if not self.enabled:
            return 0.0
        worked = max(0.0, time.monotonic() - self.t0 - self.rested)
        return 1.0 - float(np.exp(-worked / self.tau))

    def rest(self, seconds):
        """Recover: `seconds` of rest take recovery * seconds off the work time."""
        elapsed = time.monotonic() - self.t0
        self.rested = min(elapsed, self.rested + self.p.recovery * seconds)

    def factors(self):
        """{'speed', 'errors', 'think'} multipliers right now."""
        if not self.enabled:
            return {"speed": 1.0, "errors": 1.0, "think": 1.0}
        now = time.monotonic()
        a = float(np.exp(-(now - self.last) / self.p.wander_tau_s))
        self.z = a * self.z + np.sqrt(1 - a * a) * self.rng.standard_normal()
        self.last = now
        f, p = self.level(), self.p
        lerp = lambda fresh, tired: fresh + (tired - fresh) * f  # noqa: E731
        return {"speed": lerp(p.speed_fresh, p.speed_tired) * float(np.exp(p.wander_sd * self.z)),
                "errors": lerp(p.errors_fresh, p.errors_tired),
                "think": lerp(p.think_fresh, p.think_tired)}
HICK_A, HICK_B = 0.3, 0.25   # visual search: a + b*log2(N+1) seconds

# Shortest Start-search prefix that reliably ranks the app first (conservative: shorter
# ones like "wor" can match WordPad or settings pages on some machines)
SAFE_PREFIX = {"chrome": "chrom", "word": "word", "excel": "exce", "notepad": "notep"}


class Human:
    def __init__(self, persona=None, seed=None, use_model=True, hm=None, fatigue=True, **persona_overrides):
        """
        persona:  a Persona; default = Persona.sample(rng, **persona_overrides)
        seed:     reproduces the whole run (persona, paths, typing, choices)
        use_model: False forces the fallback mouse generator
        fatigue:  False keeps the person at their average pace all session
        """
        self.rng = np.random.default_rng(seed)
        self.persona = persona or Persona.sample(self.rng, **persona_overrides)
        p = self.persona
        self.hm = hm or HumanMouse(use_model=use_model, temperature=p.mouse_temperature, speed=p.mouse_speed,
                                   click_spread=p.click_spread)
        self.hm.rng = self.rng                    # one random stream for everything
        self.fatigue = Fatigue(self.rng, p.fatigue_tau_min, enabled=fatigue)
        self.f = self.fatigue.factors()
        self.keys_speed = self.f["speed"]
        self.keys = human_typing.key_timer(p.wpm * self.keys_speed, self.rng)
        self.last_device = None
        self._update()

    # --- Fatigue ------------------------------------------------------------------------

    def _update(self):
        """Apply the current fatigue / tempo factors to the mouse and key timing."""
        self.f = self.fatigue.factors()
        p = self.persona
        self.hm.speed = p.mouse_speed * self.f["speed"]
        self.hm.click_spread = p.click_spread * self.f["errors"] ** 0.5
        if abs(self.f["speed"] / self.keys_speed - 1) > 0.04:      # re-time glue keys when the pace moved
            self.keys_speed = self.f["speed"]
            self.keys = human_typing.key_timer(p.wpm * self.keys_speed, self.rng)

    @property
    def error_level(self):
        """Current error multiplier: persona's error_scale times fatigue (use for slip rates)."""
        return self.persona.error_scale * self.f["errors"]

    def rest(self, seconds):
        """Time away from the desk (idle break): partly recovers from fatigue."""
        self.fatigue.rest(seconds)
        self._update()

    def state(self):
        """Short description of the current pace, for logs."""
        f = self.f
        return (f"fatigue {self.fatigue.level():.2f}: speed x{f['speed']:.2f}, errors x{f['errors']:.2f}, "
                f"thinking x{f['think']:.2f}")

    # --- Timing -------------------------------------------------------------------------

    def think(self, kind="mental", scale=1.0):
        """Pause before an action. kind: 'mental', 'scan', 'confirm' or 'glance'."""
        median, sigma = THINK[kind]
        time.sleep(_lognormal(self.rng, median * self.persona.think_scale * self.f["think"] * scale, sigma))

    def use_device(self, device):
        """Insert homing time when the hand moves to the other device (and refresh the
        fatigue / tempo factors - every action passes through here)."""
        self._update()
        if self.last_device is not None and self.last_device != device:
            time.sleep(_lognormal(self.rng, *HOMING))
        self.last_device = device

    def prefers_keyboard(self, bias=0.0):
        """Decide, for one action, whether this person uses the keyboard way."""
        return self.rng.random() < float(np.clip(self.persona.shortcut_pref + bias, 0.03, 0.97))

    def choose(self, items, top_n=8, search=True):
        """
        Pick one of `items` (ordered top to bottom) like a person scanning a list:
        upper items are likelier, and the search takes longer the more there are.
        """
        if not items:
            return None
        candidates = items[:top_n]
        if search:
            self.visual_search(len(candidates))
        weights = 1.0 / (1 + np.arange(len(candidates)))
        return candidates[self.rng.choice(len(candidates), p=weights / weights.sum())]

    def visual_search(self, n):
        """Time to find one target among n candidates (Hick-Hyman)."""
        t = (HICK_A + HICK_B * np.log2(n + 1)) * self.persona.think_scale * self.f["think"]
        time.sleep(_lognormal(self.rng, t, 0.3))

    # --- Mouse --------------------------------------------------------------------------

    def move_to(self, x, y):
        self.use_device("mouse")
        self.hm.move_to(x, y)

    def click_rect(self, rect, button="left"):
        self.use_device("mouse")
        return self.hm.click_rect(rect, button)

    def click_element(self, ctrl, button="left", modifiers=()):
        """Click a UIA control; its position is re-checked just before the click.
        modifiers: keys held during the click, e.g. ('ctrl',)."""
        self.use_device("mouse")
        return self.hm.click_element(ctrl, button, modifiers)

    def double_click_rect(self, rect):
        self.use_device("mouse")
        self.hm.double_click_rect(rect)

    def click_field(self, rect, p_park=0.35):
        """
        Click into a text field. Sometimes the cursor is then moved a little aside
        so it doesn't cover the text that is about to be typed.
        """
        point = self.click_rect(rect)
        if self.rng.random() < p_park:
            self.park_cursor(rect)
        return point

    def park_cursor(self, avoid_rect):
        """Move the cursor a short distance out of `avoid_rect` (below it, or beside it)."""
        x0, _ = mouse.get_position()
        left, top, right, bottom = screen.virtual_screen_rect()
        x = x0 + self.rng.normal(0, 60)
        y = avoid_rect[3] + self.rng.uniform(25, 140)
        if y > bottom - 10:                      # no room below: go above instead
            y = avoid_rect[1] - self.rng.uniform(25, 100)
        x = float(np.clip(x, left + 10, right - 10))
        y = float(np.clip(y, top + 10, bottom - 10))
        self.hm.move_to(int(x), int(y))

    # --- Keyboard -----------------------------------------------------------------------

    def press(self, key, presses=1):
        self.use_device("keyboard")
        human_typing.press_like_human(key, self.rng, presses=presses, timer=self.keys)

    def hotkey(self, *keys):
        self.use_device("keyboard")
        human_typing.hotkey_like_human(*keys, rng=self.rng, timer=self.keys)

    def type(self, text, mode=None, **kw):
        """Type with the learned typing model at this persona's speed / typo rate."""
        self.use_device("keyboard")
        kw.setdefault("pause_scale", self.persona.pause_scale * self.f["think"])
        kw["error_scale"] = kw.get("error_scale", 1.0) * self.error_level
        if mode == "compose":     # keyboard-minded people delete whole words with Ctrl+Backspace
            kw.setdefault("word_deletes", 0.8 * self.persona.shortcut_pref)
        return human_typing.type_like_human(text, rng=self.rng, wpm=self.persona.wpm * self.f["speed"],
                                            mode=mode, **kw)

    def type_short(self, text, **kw):
        """Type a short input (URL, search, app name, path): rhythm + typos, no composition."""
        return self.type(text, mode="transcribe", **kw)

    def select_all_and_type(self, text, **kw):
        self.hotkey("ctrl", "a")
        self.think("glance", 0.6)
        return self.type_short(text, **kw)

    # --- Composite actions (method chosen per call) ---------------------------------------

    def open_app(self, name, open_wait=config.START_MENU_OPEN_WAIT, search_wait=config.START_MENU_SEARCH_WAIT):
        """Open an app through Start search: press Win or click the Start button,
        type the start of its name (people stop once the app shows up), look at the
        top result, Enter."""
        start = None if self.prefers_keyboard(0.25) else _start_button()
        if start is not None:
            self.click_element(start)
        else:
            self.press("win")
        time.sleep(open_wait * self.rng.uniform(0.9, 1.3))
        self.type_short(self.search_prefix(name))
        time.sleep(search_wait * self.rng.uniform(0.9, 1.3))
        self.think("confirm")
        self.press("enter")

    def search_prefix(self, name):
        """
        How much of an app's name this person types into Start search. Known apps have a
        shortest prefix that reliably puts them on top (SAFE_PREFIX); people type that
        or a letter or two more. Unknown names are typed in full.
        """
        safe = SAFE_PREFIX.get(name.lower())
        if not safe:
            return name
        return name[:int(self.rng.integers(len(safe), len(name) + 1))]

    def maximize_window(self, hwnd):
        """Maximise the window in front: Win+Up or its Maximize button. No-op if it
        already is maximised or isn't the foreground window."""
        user32 = ctypes.windll.user32
        if user32.IsZoomed(hwnd) or user32.GetForegroundWindow() != hwnd:
            return
        button = None
        if not self.prefers_keyboard():
            button = ui_elements.find_element(auto.ControlFromHandle(hwnd), "Maximize", "ButtonControl",
                                              timeout=1, search_depth=4)
        if button is not None and ui_elements.is_visible(button):
            self.click_element(button)
        else:
            self.hotkey("win", "up")
        time.sleep(0.6)

    def place_new_window(self, hwnd):
        """A window just opened: this person maximises it (maximize_pref) or works in it as it is."""
        if self.rng.random() < self.persona.maximize_pref:
            self.think("glance")
            self.maximize_window(hwnd)

    def focus_address_bar(self):
        """Click Chrome's address bar or press Ctrl+L. Returns True if it has focus."""
        bar = None if self.prefers_keyboard() else browser.get_address_bar_rect()
        if bar:
            self.click_field(bar, p_park=0.2)
        else:
            self.hotkey("ctrl", "l")
        time.sleep(0.15)
        return browser.address_bar_has_focus()

    def navigate(self, url):
        """Go to a URL / search the way a person does. Returns False if the bar never got focus."""
        if not self.focus_address_bar():
            return False
        self.think("glance")
        self.type_short(url)
        self.think("confirm")
        self.press("enter")
        return True

    def go_back(self):
        """Browser Back: the toolbar button or Alt+Left."""
        back = None if self.prefers_keyboard() else browser.get_toolbar_button_rect("Back")
        if back:
            self.click_rect(back)
        else:
            self.hotkey("alt", "left")

    def save_as(self, path, shortcut=("f12",), dialog_wait=config.DIALOG_OPEN_WAIT, overwrite=True):
        """Save through the Save As dialog: shortcut, replace the file name, Enter."""
        path = os.path.abspath(path)
        existed = os.path.exists(path)
        if len(shortcut) > 1:
            self.hotkey(*shortcut)
        else:
            self.press(shortcut[0])
        time.sleep(dialog_wait)
        self.think("scan", 0.6)
        self.select_all_and_type(path)
        self.think("confirm")
        self.press("enter")
        if existed:
            self.answer_prompt("^Yes$" if overwrite else "^No$")

    def answer_prompt(self, name_regex, timeout=3):
        """Click a dialog button whose name matches the regex (e.g. '^Don.t Save$')."""
        for root in (ui_elements.get_foreground_window(), None):
            btn = office._find_button_regex(root, name_regex, timeout)
            if btn:
                self.think("scan", 0.7)          # read the prompt first
                self.click_element(btn)
                return True
        return False


def _start_button():
    """The taskbar Start button (UIA control), or None."""
    try:
        taskbar = auto.PaneControl(searchDepth=1, ClassName="Shell_TrayWnd")
        if not taskbar.Exists(0.5, 0.1):
            return None
        btn = ui_elements.find_element(taskbar, "Start", "ButtonControl", timeout=1)
        return btn if btn and ui_elements.is_visible(btn) else None
    except Exception:
        return None
