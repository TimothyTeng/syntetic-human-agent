"""
Offline self-test: plans many texts at random speeds and checks, WITHOUT pressing
any real keys, that

  1. the keystroke plan replays to exactly the target text, and
  2. the key-down / key-up timeline the player would send (with Shift overlap and
     rollover) produces exactly the target text on a simulated US keyboard, and
     never presses a non-shifted key while Shift is down (or vice versa).

Usage:
    python -m algorithms.typing_model.selftest [--n 300]
"""

import argparse

import numpy as np

from .keys import SHIFTED
from .planner import TypingConfig, plan_keystrokes
from .runtime import TypingModel, apply_plan

TEXTS = [
    "weather in singapore tomorrow",
    "ST Engineering annual report 2025",
    "How do I reset my password?",
    ("Hi Sarah, thanks for sending the report yesterday. I think we should review the budget numbers "
     "before the meeting on Friday. Could you also check whether the new supplier quotes are included? "
     "Let me know if you need any help with it.\n\nBest regards,\nTim"),
    ("The results were better than expected (about 12% above target), but we still need to confirm "
     "the numbers with finance. I'll send an update once I've heard back from them!"),
    "Meeting notes: 1) budget; 2) hiring - \"urgent\"; 3) Q&A @ 3pm #team",
]

_UNSHIFT = {v: k for k, v in SHIFTED.items()}


def simulate_timeline(events):
    """Apply player events to a virtual text field. Returns (text, problems)."""
    buf, cur, shift = [], 0, False
    problems = []
    for t, _, action, key in events:
        if action == "char":
            buf.insert(cur, key)
            cur += 1
        elif action == "down":
            if key == "shift":
                shift = True
            elif key == "backspace":
                if cur > 0:
                    del buf[cur - 1]
                    cur -= 1
            elif key == "left":
                if shift:
                    problems.append(f"{t:.3f}s arrow pressed while Shift is down")
                cur = max(0, cur - 1)
            elif key == "right":
                if shift:
                    problems.append(f"{t:.3f}s arrow pressed while Shift is down")
                cur = min(len(buf), cur + 1)
            elif key == "enter":
                buf.insert(cur, "\n")
                cur += 1
            elif key == "tab":
                buf.insert(cur, "\t")
                cur += 1
            else:
                if shift:
                    ch = key.upper() if key.isalpha() else _UNSHIFT.get(key, key)
                else:
                    ch = key
                buf.insert(cur, ch)
                cur += 1
        elif action == "up" and key == "shift":
            shift = False
    return "".join(buf), problems


def run(n=300, seed=0, verbose=True):
    from algorithms.human_typing import build_timeline

    model = TypingModel.load()
    if verbose:
        print("model sources:", model.sources)
    rng = np.random.default_rng(seed)
    bad_plan = bad_timeline = 0
    for i in range(n):
        text = TEXTS[i % len(TEXTS)]
        cfg = TypingConfig(wpm=float(rng.uniform(20, 110)), revision_scale=float(rng.uniform(0.5, 3)),
                           error_scale=float(rng.uniform(0.5, 3)))
        plan = plan_keystrokes(text, model, rng, cfg)
        if apply_plan(plan) != text:
            bad_plan += 1
            if verbose:
                print("PLAN MISMATCH:", repr(apply_plan(plan)))
            continue
        got, problems = simulate_timeline(build_timeline(plan, rng, model.timer))
        if got != text or problems:
            bad_timeline += 1
            if verbose:
                print("TIMELINE MISMATCH:", repr(got), problems[:3])
    if verbose:
        print(f"{n} plans: {bad_plan} plan mismatches, {bad_timeline} timeline mismatches")
    return bad_plan == 0 and bad_timeline == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    raise SystemExit(0 if run(a.n, a.seed) else 1)
