"""
Document-level actions for OpenOffice Writer, Excel and Notepad: open, new, save,
save-as, handle the "save changes?" prompt, close.

Shortcut differences to be aware of:
  Excel            : Save As dialog = F12,            Open dialog = Ctrl+F12
  OpenOffice Writer: Save As dialog = Ctrl+Shift+S,   Open dialog = Ctrl+O
                     (F12 switches numbering on and Ctrl+F12 inserts a table there!)
  Notepad(W11)     : Save As dialog = Ctrl+Shift+S,   Open dialog = Ctrl+O
"""

import os
import time

import uiautomation as auto

from . import apps, config, keyboard, ui_elements

WRITER_TITLE = "OpenOffice Writer"
EXCEL_TITLE = "Excel"
NOTEPAD_TITLE = "Notepad"


# --- Launching ---------------------------------------------------------------

def open_writer(timeout=40):
    """Open OpenOffice Writer from the Start menu; returns its window or None. (It opens
    straight into a blank 'Untitled 1' document - there is no Start screen.)"""
    return apps.open_via_start_menu("openoffice writer", wait_title=WRITER_TITLE, timeout=timeout)


def open_excel(timeout=30):
    """Open Microsoft Excel from the Start menu; returns its window or None."""
    return apps.open_via_start_menu("excel", wait_title=EXCEL_TITLE, timeout=timeout)


def open_notepad(timeout=config.DEFAULT_TIMEOUT):
    """Open Notepad from the Start menu; returns its window or None."""
    return apps.open_via_start_menu("notepad", wait_title=NOTEPAD_TITLE, timeout=timeout)


def new_blank_from_start_screen():
    """
    Excel opens on a 'Start' screen with 'Blank workbook' pre-selected - pressing
    Enter opens it. (OpenOffice Writer has no Start screen.)
    """
    keyboard.press_key("enter")


# --- Document actions --------------------------------------------------------

def new_document():
    """Ctrl+N - create a new blank document/workbook/tab in the focused app."""
    keyboard.hotkey("ctrl", "n")


def open_document(path, shortcut=("ctrl", "f12"), dialog_wait=config.DIALOG_OPEN_WAIT,
                  type_interval=0.0):
    """
    Open an existing file from inside the focused app via its Open dialog.

    shortcut: ('ctrl','f12') for Excel, ('ctrl','o') for OpenOffice Writer and Notepad.
    """
    keyboard.hotkey(*shortcut)
    time.sleep(dialog_wait)
    # The File name box has focus when the classic Open dialog appears
    keyboard.type_text(os.path.abspath(path), interval=type_interval)
    keyboard.press_key("enter")


def save_document():
    """Ctrl+S - save the current document (first save will open Save As)."""
    keyboard.hotkey("ctrl", "s")


def save_as(path, shortcut=("f12",), dialog_wait=config.DIALOG_OPEN_WAIT,
            overwrite=True, type_interval=0.0):
    """
    Save the current document to a full file path via the Save As dialog.

    path:      e.g. r'C:\\Users\\me\\Documents\\notes.odt'
    shortcut:  ('f12',) for Excel, ('ctrl','shift','s') for OpenOffice Writer and Notepad.
    overwrite: if the file exists, answer 'Yes' to the replace prompt.
    """
    path = os.path.abspath(path)
    existed = os.path.exists(path)

    keyboard.hotkey(*shortcut)
    time.sleep(dialog_wait)
    # The File name box is focused when the dialog opens - replace its text
    keyboard.select_all()
    keyboard.type_text(path, interval=type_interval)
    keyboard.press_key("enter")

    if existed:
        handle_replace_prompt(overwrite)


def handle_replace_prompt(overwrite=True, timeout=3):
    """
    Answer the 'file already exists. Do you want to replace it?' prompt.
    Returns True if a prompt was found and answered.
    """
    button = _find_button_regex(ui_elements.get_foreground_window(),
                                "^Yes$" if overwrite else "^No$", timeout)
    if button:
        ui_elements.click_element(button)
        return True
    return False


def handle_save_prompt(choice="dont_save", timeout=3):
    """
    Answer the 'Do you want to save changes?' prompt shown when closing.

    choice: 'save', 'dont_save' or 'cancel'.
    Returns True if the prompt was found and a button clicked.
    """
    patterns = {
        "save": r"^Save$",
        "dont_save": r"^(Don.t Save|Discard)$",   # Notepad / Excel: "Don't save", OpenOffice: "Discard"
        "cancel": r"^Cancel$",
    }
    fg = ui_elements.get_foreground_window()
    # Search from the foreground window first (the prompt is usually a child)
    for root in (fg, None):
        ctrl = _find_button_regex(root, patterns[choice], timeout)
        if ctrl:
            ui_elements.click_element(ctrl)
            return True
    return False


def _find_button_regex(root, pattern, timeout):
    """Find a button whose Name matches a case-insensitive regex."""
    root = root if root is not None else auto.GetRootControl()
    ctrl = auto.ButtonControl(searchFromControl=root, RegexName=f"(?i){pattern}")
    return ctrl if ctrl.Exists(maxSearchSeconds=timeout) else None


def close_document():
    """Ctrl+W - close the current document/tab but keep the app open."""
    keyboard.hotkey("ctrl", "w")


def close_app(title_substring):
    """Close the whole application window with Alt+F4."""
    return apps.close_window(title_substring)
