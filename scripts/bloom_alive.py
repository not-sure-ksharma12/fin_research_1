from pywinauto import Desktop
import pyautogui
import time

BLOOMBERG_TITLE_REGEX = ".*BLOOMBERG.*"
INTERVAL_SECONDS = 1800 # 30 minutes

print("Bloomberg keep-alive script started. Press Ctrl+C to stop.")

while True:
    try:
        windows = Desktop(backend="win32").windows(title_re=BLOOMBERG_TITLE_REGEX)
        if not windows:
            raise Exception("No Bloomberg window found.")
        window = windows[0]
        if window.is_minimized():
            window.restore()
        window.set_focus()
        time.sleep(1)  # Give it a moment to focus
        pyautogui.press('f1')
        print(f"Sent F1 key to Bloomberg window: {window.window_text()}")
    except Exception as e:
        print(f"Could not send key: {e}")
    time.sleep(INTERVAL_SECONDS)