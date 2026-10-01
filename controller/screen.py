"""
Screen / image based locating.

Use this when an element can't be found through UI Automation (ui_elements.py):
capture a small PNG of the icon/button once (Snipping Tool, Win+Shift+S), save
it in assets/icons/, then locate it on screen by template matching.

All "box" results are (left, top, width, height). Use box_center() to get a
clickable point.
"""

import ctypes
import ctypes.wintypes
import os
import time

import pyautogui

from . import config

try:
    import pytesseract  # optional OCR support
except ImportError:
    pytesseract = None


def screen_size():
    """Return the primary monitor resolution as (width, height)."""
    size = pyautogui.size()
    return size.width, size.height


def virtual_screen_rect():
    """
    Bounding rect (left, top, right, bottom) of the whole desktop across ALL
    monitors, in physical pixels. A second monitor may start at a negative x/y
    if it sits left of / above the primary one.
    """
    gsm = ctypes.windll.user32.GetSystemMetrics
    left, top = gsm(76), gsm(77)            # SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN
    return left, top, left + gsm(78), top + gsm(79)  # SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN


def screenshot(region=None, path=None):
    """
    Take a screenshot and return it as a PIL Image.

    region: optional (left, top, width, height) to capture only part of screen.
    path:   optional file path to also save the image (e.g. 'debug.png').
    """
    img = pyautogui.screenshot(region=region)
    if path:
        img.save(path)
    return img


def _resolve_icon(image):
    """Allow passing just 'save_button.png' - looked up in assets/icons/."""
    if isinstance(image, str) and not os.path.isabs(image) and not os.path.exists(image):
        candidate = os.path.join(config.ICONS_DIR, image)
        if os.path.exists(candidate):
            return candidate
    return image


def locate_image(image, confidence=0.9, region=None, grayscale=False):
    """
    Find an image (icon/button template) on screen.

    image:      path to PNG (or file name inside assets/icons/).
    confidence: 0-1 match threshold (requires opencv). Lower if not found,
                raise if you get false matches.
    region:     (left, top, width, height) to search only part of the screen
                (much faster).
    grayscale:  True speeds up matching ~30% at slight accuracy cost.

    Returns (left, top, width, height) or None if not found.
    """
    try:
        box = pyautogui.locateOnScreen(
            _resolve_icon(image), confidence=confidence, region=region, grayscale=grayscale
        )
    except pyautogui.ImageNotFoundException:
        # Newer pyscreeze raises instead of returning None
        return None
    if box is None:
        return None
    return int(box.left), int(box.top), int(box.width), int(box.height)


def locate_all_images(image, confidence=0.9, region=None, grayscale=False):
    """Return a list of every (left, top, width, height) match of the image."""
    try:
        boxes = pyautogui.locateAllOnScreen(
            _resolve_icon(image), confidence=confidence, region=region, grayscale=grayscale
        )
        return [(int(b.left), int(b.top), int(b.width), int(b.height)) for b in boxes]
    except pyautogui.ImageNotFoundException:
        return []


def wait_for_image(image, timeout=config.DEFAULT_TIMEOUT, confidence=0.9, region=None):
    """
    Poll the screen until the image appears or timeout (seconds) expires.
    Returns the box (left, top, width, height) or None on timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        box = locate_image(image, confidence=confidence, region=region)
        if box:
            return box
        time.sleep(config.POLL_INTERVAL)
    return None


def box_center(box):
    """Convert (left, top, width, height) into its centre point (x, y)."""
    left, top, width, height = box
    return left + width // 2, top + height // 2


def box_to_rect(box):
    """Convert (left, top, width, height) into (left, top, right, bottom)."""
    left, top, width, height = box
    return left, top, left + width, top + height


def pixel_color(x, y):
    """Return the (R, G, B) colour of the screen pixel at (x, y)."""
    return pyautogui.pixel(x, y)


def ocr_available():
    """True if pytesseract + the Tesseract binary are installed."""
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def locate_text_ocr(text, region=None, case_sensitive=False):
    """
    OPTIONAL fallback: find a word/phrase on screen with OCR.

    Requires `pip install pytesseract` and the Tesseract-OCR program
    (https://github.com/UB-Mannheim/tesseract/wiki). Only matches text that
    OCR splits into consecutive words.

    Returns (left, top, width, height) in screen coordinates, or None.
    """
    if not ocr_available():
        raise RuntimeError("OCR not available - install pytesseract and Tesseract-OCR")

    img = screenshot(region=region)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    offset_x, offset_y = (region[0], region[1]) if region else (0, 0)

    target = text.split()
    words = data["text"]
    norm = (lambda s: s) if case_sensitive else (lambda s: s.lower())
    target = [norm(w) for w in target]

    # Slide over OCR words looking for the target word sequence
    for i in range(len(words) - len(target) + 1):
        if all(norm(words[i + j].strip()) == target[j] for j in range(len(target))):
            last = i + len(target) - 1
            left = data["left"][i]
            top = min(data["top"][i:last + 1])
            right = data["left"][last] + data["width"][last]
            bottom = max(data["top"][k] + data["height"][k] for k in range(i, last + 1))
            return left + offset_x, top + offset_y, right - left, bottom - top
    return None


def ocr_lines(rect=None, min_conf=50):
    """
    OPTIONAL fallback: words on screen grouped into text lines by OCR.

    rect: (left, top, right, bottom) area to read; None = primary screen.
    Returns [[{'text', 'rect'}, ...], ...] (rect = (left, top, right, bottom) in
    screen pixels), top to bottom, or [] if OCR is not available.
    """
    if not ocr_available():
        return []
    region = (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]) if rect else None
    img = screenshot(region=region)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    ox, oy = (rect[0], rect[1]) if rect else (0, 0)
    lines = {}
    for i, text in enumerate(data["text"]):
        if not text.strip() or float(data["conf"][i]) < min_conf:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        l, t = data["left"][i] + ox, data["top"][i] + oy
        lines.setdefault(key, []).append({"text": text.strip(),
                                          "rect": (l, t, l + data["width"][i], t + data["height"][i])})
    ordered = sorted(lines.values(), key=lambda ws: min(w["rect"][1] for w in ws))
    return [sorted(ws, key=lambda w: w["rect"][0]) for ws in ordered]


def monitor_rects():
    """(left, top, right, bottom) of every monitor, in physical pixels. Unlike
    virtual_screen_rect(), this leaves out the dead areas between monitors of
    different sizes, where the cursor can't go."""
    rects = []

    @ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.wintypes.RECT),
                        ctypes.c_void_p)
    def _collect(_hmon, _hdc, rect, _data):
        r = rect.contents
        rects.append((r.left, r.top, r.right, r.bottom))
        return 1

    ctypes.windll.user32.EnumDisplayMonitors(None, None, _collect, 0)
    return rects


def on_screen(x, y, rects=None, margin=0):
    """True if (x, y) lies on a monitor (at least `margin` px inside its edges)."""
    for l, t, r, b in rects or monitor_rects():
        if l + margin <= x < r - margin and t + margin <= y < b - margin:
            return True
    return False


def nearest_on_screen(x, y, rects=None, margin=0):
    """The point on any monitor (at least `margin` px inside) closest to (x, y)."""
    best, best_d = (x, y), None
    for l, t, r, b in rects or monitor_rects():
        px = min(max(x, l + margin), r - 1 - margin)
        py = min(max(y, t + margin), b - 1 - margin)
        d = (px - x) ** 2 + (py - y) ** 2
        if best_d is None or d < best_d:
            best, best_d = (px, py), d
    return best
