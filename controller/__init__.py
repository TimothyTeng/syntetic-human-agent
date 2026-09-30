"""
controller - base desktop control layer.

Import the sub-modules and call their functions, e.g.:

    from controller import mouse, keyboard, apps, office, browser, screen, ui_elements

    office.open_notepad()
    keyboard.type_text("hello")
    browser.open_chrome()
    browser.navigate_to("https://en.wikipedia.org")
    browser.click_link("Random article")

See CONTROLLER.md for the full function reference.
"""

from . import config  # must be first: sets DPI awareness + pyautogui settings
from . import mouse, keyboard, screen, ui_elements, apps, office, browser

__all__ = ["config", "mouse", "keyboard", "screen", "ui_elements", "apps", "office", "browser"]
