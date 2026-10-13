"""
Session scheduler: strings activities together into one continuous working session
(e.g. the 15-minute run), the way a person's time at the desk is organised.

    session = Session(human, [Activity("browse", ws.browse), Activity("write", ws.write)], minutes=15)
    session.run()

Between activities the person
  * picks what to do next with a simple Markov chain: activities are weighted, and the
    one just done is likelier to be done again (people stay on a task for a while),
  * now and then takes an idle break - no input at all for 20 s to 3 min (reading
    something on paper, a colleague, the phone), sometimes nudging the mouse on return,
  * pauses briefly to decide what to do next.

Activities receive the seconds left so they can size themselves (reading time,
amount of text) to fit; nothing starts that needs more time than is left. A failing
activity (TaskError, or an unexpected error from the UI) is logged and the session
carries on; after `max_failures` in a row it stops. The pyautogui fail-safe and
Ctrl+C always stop it.
"""

import time
import traceback
from dataclasses import dataclass
from typing import Callable

import numpy as np


class TaskError(Exception):
    """An activity found the UI in an unexpected state and stopped safely."""


@dataclass
class Activity:
    name: str
    run: Callable[[float], None]     # run(seconds_left)
    weight: float = 1.0              # how often it is chosen
    min_seconds: float = 30.0        # not started with less time than this left


@dataclass
class SessionParams:
    p_repeat_boost: float = 2.0      # weight multiplier for the activity just done
    p_break: float = 0.25            # chance of an idle break between activities
    break_median: float = 45.0       # seconds
    break_sigma: float = 0.6
    break_range: tuple = (20.0, 180.0)
    p_nudge_after_break: float = 0.5  # small mouse movement when coming back
    max_failures: int = 3            # consecutive failed activities before stopping
    drop_after: int = 2              # an activity failing this often in a row is dropped
                                     # (e.g. the app isn't installed)


class Session:
    def __init__(self, human, activities, minutes=15.0, params=SessionParams(), log=print):
        self.human = human
        self.activities = activities
        self.seconds = minutes * 60
        self.params = params
        self.log = log
        self.history = []            # (name, seconds, ok)
        self.fails = {}              # activity name -> consecutive failures

    def run(self):
        """Run until the time is up. Returns the history [(activity, seconds, ok), ...]."""
        p, rng = self.params, self.human.rng
        end = time.monotonic() + self.seconds
        prev, failures, rested = None, 0, False
        while (left := end - time.monotonic()) > 5:
            if prev is not None and not rested and rng.random() < p.p_break:
                self.idle(min(self._break_length(), left))
                rested = True                     # back from a break: do something next
                continue
            rested = False
            act = self._choose(prev, left)
            if act is None:                       # nothing fits in the time left
                self.idle(left)
                break
            pace = f" | {self.human.state()}" if hasattr(self.human, "state") else ""
            self.log(f"--- {act.name} ({left / 60:.1f} min left){pace}")
            self.human.think("mental")            # deciding / getting started
            t0 = time.monotonic()
            try:
                act.run(end - time.monotonic())
                failures, ok = 0, True
                self.fails[act.name] = 0
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:              # TaskError, or an unexpected UI / COM error
                if type(exc).__name__ == "FailSafeException":
                    raise                         # the user's emergency stop always wins
                where = "" if isinstance(exc, TaskError) else                     f" ({type(exc).__name__} at {traceback.extract_tb(exc.__traceback__)[-1].name})"
                self.log(f"{act.name} stopped: {exc}{where}")
                failures, ok = failures + 1, False
                self.fails[act.name] = self.fails.get(act.name, 0) + 1
                if self.fails[act.name] >= p.drop_after:
                    self.log(f"{act.name} failed {p.drop_after} times in a row - not doing it again this session")
            self.history.append((act.name, time.monotonic() - t0, ok))
            if failures >= p.max_failures:
                self.log(f"{failures} activities failed in a row - ending the session")
                break
            prev = act
        self.log("Session finished: " + self.summary())
        return self.history

    def _choose(self, prev, left):
        fits = [a for a in self.activities
                if a.min_seconds <= left and self.fails.get(a.name, 0) < self.params.drop_after]
        if not fits:
            return None
        w = np.array([a.weight * (self.params.p_repeat_boost if a is prev else 1.0) for a in fits])
        return fits[self.human.rng.choice(len(fits), p=w / w.sum())]

    def _break_length(self):
        p = self.params
        v = p.break_median * np.exp(self.human.rng.normal(0, p.break_sigma))
        return float(np.clip(v, *p.break_range))

    def idle(self, seconds):
        """No input for `seconds` (the person is away from the mouse and keyboard)."""
        self.log(f"--- idle break ({seconds:.0f} s, no input)")
        nudge = self.human.rng.random() < self.params.p_nudge_after_break and seconds > 5
        time.sleep(max(0.0, seconds - (1.5 if nudge else 0.0)))
        if hasattr(self.human, "rest"):
            self.human.rest(seconds)              # a break partly recovers from fatigue
        if nudge:                                 # hand back on the mouse
            self.human.hm.drift(max_dist=60)
            self.human.last_device = "mouse"
        self.history.append(("idle", seconds, True))

    def summary(self):
        by = {}
        for name, secs, ok in self.history:
            n, s, fails = by.get(name, (0, 0.0, 0))
            by[name] = (n + 1, s + secs, fails + (not ok))
        return ", ".join(f"{k} x{n} ({s / 60:.1f} min{f', {f} failed' if f else ''})" for k, (n, s, f) in by.items())
