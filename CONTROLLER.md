# Base Controller — Function Reference

The `controller` package is the **base control layer** for the Synthetic Human Agent. It holds small, single-purpose functions for mouse, keyboard, launching apps, saving files, locating buttons/icons and driving Chrome. Your behaviour/timing algorithm sits on top and calls these. See `ALGORITHMS.md` for the behaviour layer.

**Timing.** The controller adds **no randomness and almost no delays**: `pyautogui.PAUSE = 0`. Every function that moves, types or scrolls takes explicit `duration` / `interval` parameters, so your entropy layer decides all the pacing. The only fixed waits are short "UI settle" pauses needed for correctness (for example, waiting for the Start menu to open). They live in `controller/config.py` and can be overridden per call.

**How apps are launched.** Programs are opened the way a person would open them: through the Start menu search or File Explorer. This keeps launches going through the normal Windows shell.

---

## Setup

```bash
pip install -r requirements.txt
```

| Module | Why it's needed |
|---|---|
| `pyautogui` | Mouse, keyboard, screenshots, image template matching. Also installs `pygetwindow` (window find/focus) and `pyscreeze`. |
| `uiautomation` | Windows UI Automation: finds buttons, menus and links **by name** and returns exact screen coordinates. |
| `pyperclip` | Clipboard read/write, used to type non-ASCII characters. |
| `opencv-python` | Enables `confidence=` fuzzy image matching. |
| `pillow` | Screenshots. |
| `psutil` | Checks whether a process is running. |
| `pytesseract` *(optional)* | OCR fallback for finding on-screen text. It also needs the [Tesseract-OCR installer](https://github.com/UB-Mannheim/tesseract/wiki). |

## Quick start

```python
from controller import mouse, keyboard, apps, office, browser, screen, ui_elements

office.open_notepad()
keyboard.type_text("Meeting notes\n", interval=0.05)
office.save_as(r"C:\Users\me\Documents\notes.txt", shortcut=("ctrl", "shift", "s"))
office.close_app("Notepad")

browser.open_chrome()
browser.navigate_to("https://en.wikipedia.org")
browser.wait_for_page_load("Wikipedia")
browser.scroll_page(-5)
browser.click_link("Random article", duration=0.6)
browser.close_chrome()
```

Run the smoke test from the project root. Don't touch the mouse or keyboard while it runs:

```bash
python -m examples.demo_basic
```

**Emergency stop:** slam the mouse into any screen corner. `pyautogui.FAILSAFE` then raises an exception and the script stops.

### Coordinate conventions
- **Point:** `(x, y)`, in screen pixels from the top-left of the primary monitor.
- **rect:** `(left, top, right, bottom)`. Returned by `ui_elements` and `browser`.
- **box:** `(left, top, width, height)`. Returned by `screen` image/OCR functions. Convert it with `screen.box_center()` or `screen.box_to_rect()`.
- The process is set DPI-aware, so all coordinates are physical pixels even with Windows display scaling.

---

## `controller/mouse.py`

| Function | Description |
|---|---|
| `get_position()` | Returns the current cursor `(x, y)`. |
| `move_to(x, y, duration=0, tween=linear)` | Moves the cursor to absolute `(x, y)` over `duration` seconds. `tween` is a pyautogui easing function, e.g. `pyautogui.easeInOutQuad`. |
| `move_relative(dx, dy, duration=0, tween=linear)` | Moves the cursor by an offset. |
| `click(x=None, y=None, button='left', duration=0)` | Single click. Moves to `(x, y)` first if given. `button` is `'left'`, `'right'` or `'middle'`. |
| `double_click(x=None, y=None, button='left', duration=0, interval=0)` | Double click. `interval` is the gap between the two clicks. |
| `right_click(x=None, y=None, duration=0)` | Right click (context menu). |
| `mouse_down(button='left')` / `mouse_up(button='left')` | Press or release a button, for custom drags and human-timed clicks. |
| `drag_to(x, y, duration=0.5, button='left')` | Click-drag from the current position to `(x, y)`, e.g. to select text. |
| `scroll(amount, x=None, y=None)` | Vertical wheel scroll in **notches** (1 notch = 120 wheel delta ≈ 100 px in a browser). Positive = up, negative = down. Fractions allowed. |
| `hscroll(amount, x=None, y=None)` | Horizontal scroll in notches. Positive = right. |
| `click_center_of(rect, button='left', duration=0)` | Clicks the centre of a `(left, top, right, bottom)` rect. |

For custom human-like paths, have your algorithm generate waypoints and call `move_to(x, y)` for each one. `algorithms/player.py` does exactly this.

## `controller/keyboard.py`

Key names follow pyautogui: `'enter'`, `'tab'`, `'esc'`, `'backspace'`, `'ctrl'`, `'shift'`, `'alt'`, `'win'`, `'f1'`–`'f12'`, arrows, `'pageup'`/`'pagedown'`, and single characters.

| Function | Description |
|---|---|
| `type_text(text, interval=0)` | Types a string with a **fixed** gap between keys. `\n` is typed as Enter; non-ASCII characters are pasted. |
| `type_char(ch)` | Types one character. **Use this for custom typing rhythms and typos**: loop over characters with your own delays. |
| `press_key(key, presses=1, interval=0)` | Presses a key one or more times. |
| `hotkey(*keys, interval=0)` | Key combo, e.g. `hotkey('ctrl', 's')`. |
| `key_down(key)` / `key_up(key)` | Holds or releases a key. |
| `backspace(n=1, interval=0)` | Presses Backspace n times, e.g. to fix a typo. |
| `select_all()` / `copy()` / `paste()` | Ctrl+A / Ctrl+C / Ctrl+V. |
| `set_clipboard(text)` / `get_clipboard()` | Writes or reads the clipboard. |
| `paste_text(text)` | Puts text on the clipboard and pastes it. |

Example of a typo made and then corrected, using only base functions:

```python
for ch in "Helo": keyboard.type_char(ch)
keyboard.backspace(1)
keyboard.type_text("lo")
```

## `controller/screen.py` — image and OCR locating

Use this when UI Automation can't see an element, such as desktop icons, canvas content or images. Capture a small PNG of the icon once (Win+Shift+S) and save it in `assets/icons/`.

| Function | Description |
|---|---|
| `screen_size()` | Returns `(width, height)` of the primary monitor. |
| `screenshot(region=None, path=None)` | Returns a PIL image. Pass `path` to also save it. `region` is a box. |
| `locate_image(image, confidence=0.9, region=None, grayscale=False)` | Finds a template on screen and returns a box, or `None`. `image` can be a bare file name in `assets/icons/`. |
| `locate_all_images(...)` | Returns every match as a list of boxes. |
| `wait_for_image(image, timeout=15, confidence=0.9, region=None)` | Polls until the image appears. Returns a box or `None`. |
| `box_center(box)` / `box_to_rect(box)` | Converts a box to a point or a rect. |
| `pixel_color(x, y)` | Returns the `(R, G, B)` colour at a pixel. |
| `ocr_available()` | True if pytesseract and Tesseract are installed. |
| `locate_text_ocr(text, region=None, case_sensitive=False)` | *Optional.* Finds on-screen text by OCR and returns a box, or `None`. |

```python
box = screen.locate_image("recycle_bin.png", confidence=0.85)
if box:
    mouse.double_click(*screen.box_center(box), duration=0.4)
```

## `controller/ui_elements.py` — find buttons and controls by name

This uses the Windows accessibility tree, the same data screen readers use. It works for the Office ribbon, Save/Open dialogs, Notepad, Explorer and Chrome. **It is preferred over image matching** because it is resolution-independent and returns exact rectangles.

Common `control_type` values: `'ButtonControl'`, `'EditControl'`, `'HyperlinkControl'`, `'MenuItemControl'`, `'TabItemControl'`, `'ListItemControl'`, `'TextControl'`, `'DocumentControl'`, `'CheckBoxControl'`, `'ComboBoxControl'`.

| Function | Description |
|---|---|
| `get_window(title_substring, timeout=15)` | Returns the top-level window whose title contains the text (a UIA control), or `None`. |
| `get_foreground_window()` | Returns the UIA control of the focused window. |
| `find_element(window, name, control_type=None, partial=False, timeout=5)` | Finds one element by accessible name. `window=None` searches the whole desktop. Returns a generic control or `None`; use `ctrl.GetPattern(auto.PatternId.ValuePattern)` etc. for patterns. |
| `find_elements(window, control_type=None, name_contains=None, visible_only=True, max_depth=40)` | Returns a list of all matching elements. |
| `wait_for_element(window, name, control_type=None, partial=False, timeout=15)` | Same as `find_element`, with a longer default timeout. |
| `is_visible(ctrl)` | True if the element is on screen and has non-zero size. |
| `element_rect(ctrl)` | Returns `(left, top, right, bottom)`. |
| `element_center(ctrl)` | Returns the centre `(x, y)`. |
| `click_element(ctrl, duration=0, button='left')` | Moves the mouse to the element and clicks it. |
| `list_elements(window, control_type=None)` | **Debug:** prints the name, type and rect of every element, so you can discover names. |
| `list_buttons(window)` | **Debug:** prints every visible button. |

```python
word = ui_elements.get_window("Word")
ui_elements.list_buttons(word)                         # discover names first
bold = ui_elements.find_element(word, "Bold", "ButtonControl")
ui_elements.click_element(bold, duration=0.3)
```

## `controller/apps.py` — launch, focus and close programs

| Function | Description |
|---|---|
| `open_via_start_menu(app_name, wait_title=None, timeout=15, open_wait=0.8, search_wait=1.0, type_interval=0)` | Win key → type the name → Enter. Returns the window if `wait_title` is given. |
| `open_file_via_explorer(path, wait_title=None, ...)` | Win+E → types the path into the address bar → Enter, which opens the file in its default app. The Explorer window is left open. |
| `is_running(process_name)` | E.g. `is_running('chrome.exe')`. |
| `find_windows(title_substring)` | Returns a list of matching pygetwindow windows. |
| `wait_for_window(title_substring, timeout=15)` | Waits for a window to appear. Returns it or `None`. |
| `get_active_window_title()` | Returns the title of the focused window. |
| `focus_window(title_substring)` | Brings a window to the front. Returns True/False. |
| `maximize_window(title)` / `minimize_window(title)` | Maximizes or minimizes the window. |
| `get_window_rect(title_substring)` | Returns `(left, top, right, bottom)` of the window. |
| `close_window(title_substring=None)` | Focuses the window (if a title is given), then presses Alt+F4. |

## `controller/office.py` — Word, Excel and Notepad document actions

| App | Save As | Open dialog |
|---|---|---|
| Word / Excel | `F12` (default) | `Ctrl+F12` (default) |
| Notepad (Win 11) | `shortcut=('ctrl','shift','s')` | `shortcut=('ctrl','o')` |

| Function | Description |
|---|---|
| `open_word(timeout=30)` / `open_excel(timeout=30)` / `open_notepad()` | Launch via the Start menu and return the window. |
| `new_blank_from_start_screen()` | Presses Enter on the Word/Excel start screen to open a blank document. |
| `new_document()` | Ctrl+N. |
| `open_document(path, shortcut=('ctrl','f12'), ...)` | Opens a file through the app's Open dialog. |
| `save_document()` | Ctrl+S. |
| `save_as(path, shortcut=('f12',), overwrite=True, ...)` | Opens Save As, types the full path, presses Enter, and answers "Replace?" if the file exists. |
| `handle_replace_prompt(overwrite=True, timeout=3)` | Clicks Yes or No on the "already exists" prompt. |
| `handle_save_prompt(choice='dont_save', timeout=3)` | Answers the "Save changes?" prompt on close. `choice` is `'save'`, `'dont_save'` or `'cancel'`. |
| `close_document()` | Ctrl+W: closes the document but keeps the app open. |
| `close_app(title_substring)` | Alt+F4 on the app window. |

Example Word flow:

```python
office.open_word()
time.sleep(2)
office.new_blank_from_start_screen()
keyboard.type_text("Quarterly summary\n")
office.save_as(r"C:\Users\me\Documents\summary.docx")
office.close_app("Word")
```

## `controller/browser.py` — Chrome

| Function | Description |
|---|---|
| `open_chrome(timeout=15)` | Opens Chrome from the Start menu. Waits until a **new** Chrome window is in front: a browser window or the profile picker. Already-open Chrome windows don't count. Returns the window's UIA control or `None`. |
| `get_chrome_window(timeout=5)` | Returns the Chrome **browser** window to work with: the foreground one, else the first visible one. Minimized windows and the profile picker are ignored. |
| `wait_for_browser_window(timeout=15)` | Waits until a normal browser window is in front, e.g. after choosing a profile. |
| `focus_chrome()` | Brings the Chrome browser window to the front, restoring it if minimized. |
| `maximize_chrome()` | Maximizes the current Chrome browser window. |
| `close_chrome()` | Ctrl+Shift+W: closes the Chrome window. |
| **Profile picker** ("Who's using Chrome?") | |
| `get_profile_picker(timeout=0)` | Returns the picker window if it is open, else `None`. |
| `list_profiles(timeout=5)` | Returns the profile cards left to right: `{'name', 'account', 'index', 'rect', 'click_rect', 'center'}`. `name` is the top label (e.g. `'Work'`) and `account` the bottom one (e.g. `'Timothy Teng'`). `click_rect` covers the avatar area and avoids the renameable name strip at the top. |
| `find_profile(name, index=0, exact=False)` | Matches profile **or** account name, case-insensitive; exact matches are preferred, then substrings. `index` picks among duplicates, e.g. the 2nd "Timothy" is `index=1`. |
| `select_profile(name, index=0, exact=False, duration=0)` | Finds the profile and clicks it. Follow with `wait_for_browser_window()`. |
| `get_guest_mode_rect()` | Rect of the picker's "Guest mode" button. |
| `navigate_to(url, type_interval=0)` | Ctrl+L → types the URL → Enter. |
| `get_current_url()` | Reads the address bar text. |
| `get_page_title()` | Returns the active tab title. |
| `wait_for_page_load(title_hint=None, timeout=15)` | Waits until the toolbar shows "Reload" (not "Stop") and, optionally, the title contains the hint. |
| `go_back()` / `go_forward()` / `refresh()` | Alt+Left / Alt+Right / F5. |
| `get_address_bar_rect()` | Rect of the address bar, e.g. to move the mouse there and click it. |
| `get_toolbar_button_rect(name)` | Rect of a toolbar button by name: `'Back'`, `'Forward'`, `'Reload'`, ... |
| `get_page_rect()` | Rect of the web page content area, below the toolbar. |
| `new_tab()` / `close_tab()` / `switch_tab(n)` / `next_tab()` | Tab control. |
| `scroll_page(amount, x=None, y=None)` | Wheel scroll over the page, in notches. Negative = down. |
| `page_down()` / `page_up()` | Keyboard paging. |
| `find_links(visible_only=True)` | Returns links as a list of dicts: `{'name', 'rect', 'center', 'control'}`. |
| `find_link(text, exact=False, visible_only=True)` | Returns the first link whose text matches (case-insensitive). |
| `click_link(text, exact=False, duration=0)` | Finds a visible link, moves to it and clicks. Returns True/False. |
| `scroll_until_link_visible(text, step=-3, max_scrolls=15, pause=0.4)` | Scrolls until the link is in view and returns it. |

```python
links = browser.find_links()
for l in links[:5]:
    print(l["name"], l["center"])
mouse.click(*links[0]["center"], duration=0.5)
```

**Note on link discovery:** Chrome only exposes page content (links) to UI Automation when the page is rendered and an accessibility client asks for it. `find_links()` sends that request itself and retries, but **Chrome must be visible in the foreground**. A minimized or fully covered Chrome window returns no links. If a site still returns nothing, for example a canvas-heavy page, fall back to `screen.locate_text_ocr()` or `screen.locate_image()`.

---

## Additional functions added beyond the original request

- **`wait_for_window`, `wait_for_element`, `wait_for_image`, `wait_for_page_load`**: your algorithm must not act before the UI is ready. Without these, clicks land on nothing.
- **`list_elements` / `list_buttons` / `find_links`**: debug helpers for discovering the exact names of buttons and links on *your* machine, instead of hard-coding coordinates.
- **`handle_replace_prompt` / `handle_save_prompt`**: dialogs pop up during save and close and would otherwise block the run.
- **Clipboard helpers** (`set_clipboard`, `paste_text`): needed for non-ASCII text, and useful for "copy from web → paste into Word" tasks.
- **`get_current_url`, `get_page_title`, `is_running`, `get_window_rect`**: state checks your algorithm can branch on.
- **`get_address_bar_rect`, `get_toolbar_button_rect`, `get_page_rect`**: give the behaviour layer real on-screen targets to move the mouse to.

## Suggested next building blocks (not yet implemented)

- **Email:** open Outlook (`open_via_start_menu('outlook')`), then use `ui_elements` to find "New Email", To, Subject and Send. This adds to the *Action Complexity* score.
- **Downloads:** click a download link, then use `apps.wait_for_window` or a check on the Downloads folder.
- **Excel cell navigation:** `hotkey('ctrl','g')` (Go To) to jump to a cell, then type values and press Tab/Enter.
- **Logging:** a small action logger (timestamp + action) so you can line your run up against the Sysmon / Event Log timeline when evaluating.
