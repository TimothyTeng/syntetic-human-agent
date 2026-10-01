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
"""

import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import uiautomation as auto

from controller import browser, config, mouse, office, screen, ui_elements

from . import human_typing
from .human_mouse import HumanMouse


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
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(p, k, float(v))
        return p

    def describe(self):
        return ", ".join(f"{k}={v:.2f}" for k, v in asdict(self).items())


# (median s, log-normal sigma) of the pause before an action, by kind
THINK = {
    "mental": (1.2, 0.5),    # KLM M: deciding what to do next (mean ~1.35 s)
    "scan": (0.8, 0.5),      # looking over a new screen before acting
    "confirm": (0.4, 0.4),   # checking what was typed / selected before Enter
    "glance": (0.25, 0.4),   # a quick look
}
HOMING = (0.36, 0.3)         # KLM H: hand between mouse and keyboard (mean ~0.4 s)
HICK_A, HICK_B = 0.3, 0.25   # visual search: a + b*log2(N+1) seconds


class Human:
    def __init__(self, persona=None, seed=None, use_model=True, hm=None, **persona_overrides):
        """
        persona:  a Persona; default = Persona.sample(rng, **persona_overrides)
        seed:     reproduces the whole run (persona, paths, typing, choices)
        use_model: False forces the fallback mouse generator
        """
        self.rng = np.random.default_rng(seed)
        self.persona = persona or Persona.sample(self.rng, **persona_overrides)
        p = self.persona
        self.hm = hm or HumanMouse(use_model=use_model, temperature=p.mouse_temperature, speed=p.mouse_speed)
        self.hm.rng = self.rng                    # one random stream for everything
        self.keys = human_typing.key_timer(p.wpm, self.rng)
        self.last_device = None

    # --- Timing -------------------------------------------------------------------------

    def think(self, kind="mental", scale=1.0):
        """Pause before an action. kind: 'mental', 'scan', 'confirm' or 'glance'."""
        median, sigma = THINK[kind]
        time.sleep(_lognormal(self.rng, median * self.persona.think_scale * scale, sigma))

    def _use(self, device):
        """Insert homing time when the hand moves to the other device."""
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
        t = (HICK_A + HICK_B * np.log2(n + 1)) * self.persona.think_scale
        time.sleep(_lognormal(self.rng, t, 0.3))

    # --- Mouse --------------------------------------------------------------------------

    def move_to(self, x, y):
        self._use("mouse")
        self.hm.move_to(x, y)

    def click_rect(self, rect, button="left"):
        self._use("mouse")
        return self.hm.click_rect(rect, button)

    def click_element(self, ctrl, button="left"):
        return self.click_rect(ui_elements.element_rect(ctrl), button)

    def double_click_rect(self, rect):
        self._use("mouse")
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
        self._use("keyboard")
        human_typing.press_like_human(key, self.rng, presses=presses, timer=self.keys)

    def hotkey(self, *keys):
        self._use("keyboard")
        human_typing.hotkey_like_human(*keys, rng=self.rng, timer=self.keys)

    def type(self, text, mode=None, **kw):
        """Type with the learned typing model at this persona's speed / typo rate."""
        self._use("keyboard")
        kw.setdefault("pause_scale", self.persona.pause_scale)
        kw["error_scale"] = kw.get("error_scale", 1.0) * self.persona.error_scale
        return human_typing.type_like_human(text, rng=self.rng, wpm=self.persona.wpm, mode=mode, **kw)

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
        type the name, look at the top result, Enter."""
        start = None if self.prefers_keyboard(0.25) else _start_button()
        if start is not None:
            self.click_element(start)
        else:
            self.press("win")
        time.sleep(open_wait * self.rng.uniform(0.9, 1.3))
        self.type_short(name)
        time.sleep(search_wait * self.rng.uniform(0.9, 1.3))
        self.think("confirm")
        self.press("enter")

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
