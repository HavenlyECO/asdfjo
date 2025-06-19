# hotkeys.py – global F-key controls for poker assistant (rev-C, 2025-06-18)
# =========================================================================

from __future__ import annotations
import threading
import time
import sys

# --------------------------------------------------------------
#  Optional keyboard module (requires admin on macOS/Linux)
# --------------------------------------------------------------
try:
    import keyboard  # type: ignore
except ImportError as err:
    print("[Hotkey] 'keyboard' module not available:", err, flush=True)
    print("          → Hotkeys disabled.  Install with `pip install keyboard` "
          "and run with admin privileges if needed.", flush=True)
    keyboard = None  # noqa: N816

# --------------------------------------------------------------
#  Globals
# --------------------------------------------------------------
ASSISTANT_MODES = ["ICM", "GTO", "Exploit", "Standard"]
_mode_lock = threading.Lock()
_mode_idx: int = 0

stop_evt = threading.Event()
_DEBOUNCE = 0.30
_last_press = {"f1": 0.0, "f2": 0.0, "f12": 0.0}

# --------------------------------------------------------------
#  Helpers
# --------------------------------------------------------------
def _debounced(key: str) -> bool:
    now = time.time()
    if now - _last_press[key] < _DEBOUNCE:
        return False
    _last_press[key] = now
    return True

# --------------------------------------------------------------
#  Hotkey callbacks
# --------------------------------------------------------------
def reparse_pipeline() -> None:
    if not _debounced("f1"):
        return
    print("[Hotkey] F1 → forcing manual re-parse of the screen.", flush=True)
    # TODO: insert pipeline refresh hook
    print("Re-parse complete.", flush=True)

def switch_assistant_mode() -> str:
    global _mode_idx
    if not _debounced("f2"):
        return current_mode()
    with _mode_lock:
        _mode_idx = (_mode_idx + 1) % len(ASSISTANT_MODES)
        mode = ASSISTANT_MODES[_mode_idx]
    print(f"[Hotkey] F2 → assistant mode switched to: {mode}", flush=True)
    # TODO: notify HUD/engine
    return mode

def stop_script() -> None:
    if not _debounced("f12"):
        return
    print("[Hotkey] F12 → stopping script …", flush=True)
    stop_evt.set()

# --------------------------------------------------------------
#  Public API
# --------------------------------------------------------------
def current_mode() -> str:
    with _mode_lock:
        return ASSISTANT_MODES[_mode_idx]

def hotkey_listener() -> None:
    if keyboard is None:
        return
    keyboard.add_hotkey("f1", reparse_pipeline)
    keyboard.add_hotkey("f2", switch_assistant_mode)
    keyboard.add_hotkey("f12", stop_script)
    print("[Hotkey] Enabled  F1=Re-parse | F2=Mode | F12=Stop", flush=True)
    keyboard.wait()             # keep thread alive

def start_hotkey_thread() -> threading.Thread | None:
    """
    Launch the listener in a daemon thread.
    Returns the Thread object (or None if keyboard hooks unavailable).
    """
    if keyboard is None:
        return None
    t = threading.Thread(target=hotkey_listener, name="HotkeyThread", daemon=True)
    t.start()
    return t

# --------------------------------------------------------------
#  Demo main loop
# --------------------------------------------------------------
if __name__ == "__main__":
    if keyboard is None:
        print("Keyboard hooks unavailable; exiting demo.", flush=True)
        sys.exit(0)

    start_hotkey_thread()

    try:
        while not stop_evt.is_set():
            print(f"Current assistant mode: {current_mode()}", flush=True)
            stop_evt.wait(5.0)  # wait or exit early
        print("Script stopped by user.", flush=True)
    except KeyboardInterrupt:
        print("Interrupted — exiting.", flush=True)

