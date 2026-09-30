"""
Smoke test for the controller package. Exercises each module once.

Run from the project root:   python -m examples.demo_basic
Don't touch the mouse/keyboard while it runs.
ABORT at any time by slamming the mouse into a screen corner.
"""

import os
import tempfile
import time

from controller import apps, browser, keyboard, mouse, office, screen, ui_elements


def demo_info():
    print("Screen size:", screen.screen_size())
    print("Mouse position:", mouse.get_position())
    print("Active window:", apps.get_active_window_title())


def demo_notepad():
    out_path = os.path.join(tempfile.gettempdir(), "controller_demo.txt")
    if os.path.exists(out_path):
        os.remove(out_path)

    print("Opening Notepad...")
    if not office.open_notepad():
        print("  Notepad window not found")
        return
    time.sleep(1)
    office.new_document()          # fresh tab so we don't type into an old one
    time.sleep(0.5)
    keyboard.type_text("Hello from the base controller.\nSecond line.", interval=0.03)

    print("Saving to", out_path)
    office.save_as(out_path, shortcut=("ctrl", "shift", "s"))
    time.sleep(1.5)
    print("  File exists:", os.path.exists(out_path))

    office.close_app(office.NOTEPAD_TITLE)
    time.sleep(1)
    office.handle_save_prompt("dont_save", timeout=1)   # in case other tabs are dirty


def demo_chrome():
    print("Opening Chrome...")
    if not browser.open_chrome():
        print("  Chrome window not found")
        return
    time.sleep(1.5)
    if browser.get_profile_picker(timeout=1):        # "Who's using Chrome?" screen
        first = browser.list_profiles()[0]
        print("  Profile picker - choosing", first["name"])
        mouse.click(*first["center"], duration=0.4)
        browser.wait_for_browser_window()
        time.sleep(1.0)
    browser.navigate_to("https://en.wikipedia.org/wiki/Main_Page", type_interval=0.02)
    print("  Loaded:", browser.wait_for_page_load(title_hint="Wikipedia"))
    print("  URL:", browser.get_current_url())

    links = browser.find_links()
    print(f"  {len(links)} visible links, first 10:")
    for link in links[:10]:
        print("   ", repr(link["name"]), link["center"])

    browser.scroll_page(-5)
    time.sleep(1)
    browser.scroll_page(5)
    time.sleep(0.5)

    # Click the first link that has some text
    target = next((l for l in links if len(l["name"]) > 3), None)
    if target:
        print("  Clicking:", target["name"])
        mouse.click(*target["center"], duration=0.4)
        browser.wait_for_page_load()
        print("  Now on:", browser.get_page_title())

    print("  Chrome toolbar buttons:")
    demo_ui_listing()
    time.sleep(2)
    browser.close_chrome()


def demo_ui_listing():
    """Print the buttons of the Chrome toolbar - shows how to discover names."""
    win = browser.get_chrome_window(timeout=2)
    if win:
        ui_elements.list_buttons(win)


if __name__ == "__main__":
    print("Starting in 3 seconds...")
    time.sleep(3)
    demo_info()
    demo_notepad()
    demo_chrome()
    print("Done.")
