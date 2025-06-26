"""Thin client that obtains screenshots from `screenshot_capture_server.py` and
forwards them to the main assistant server."""

import io
import time
import threading
import queue
import requests
from tkinter import Tk, Label, StringVar
import pyttsx3


# --- CONFIGURATION ---
SERVER_URL = "http://24.199.98.206:5000/api/advice"
CAPTURE_SERVER_URL = "http://localhost:5002/api/capture"
API_KEY = "your-secure-api-key"
TARGET_FPS = 32
GUI_UPDATE_INTERVAL = 0.1
ENABLE_TTS = False
FRAME_INTERVAL = 1.0 / TARGET_FPS


# --- Speech Engine Management (same as local_client) ---
speech_queue = queue.Queue()
speech_engine = None
speech_thread = None
speech_running = False


def speech_worker():
    global speech_engine, speech_running
    speech_engine = pyttsx3.init()
    speech_running = True
    while speech_running:
        try:
            text = speech_queue.get(timeout=0.5)
            if text is None:
                break
            speech_engine.say(text)
            speech_engine.runAndWait()
            speech_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"TTS Error: {e}")
            time.sleep(0.1)


def speak(text):
    if not ENABLE_TTS:
        return
    global speech_thread
    if speech_thread is None or not speech_thread.is_alive():
        speech_thread = threading.Thread(target=speech_worker, daemon=True)
        speech_thread.start()
    speech_queue.put(text)


# --- Overlay UI ---
class OverlayWindow:
    def __init__(self):
        self.root = Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.geometry("+100+100")
        self.var = StringVar()
        self.label = Label(
            self.root,
            textvariable=self.var,
            font=("Arial", 28),
            bg="yellow",
            fg="black",
            padx=10,
            pady=10,
        )
        self.label.pack()
        self.var.set("Connecting to server...")
        self.last_update = 0

    def set_text(self, txt: str):
        now = time.time()
        if now - self.last_update >= GUI_UPDATE_INTERVAL:
            self.var.set(txt)
            self.root.update()
            self.last_update = now

    def show(self):
        self.root.deiconify()
        self.root.update()

    def hide(self):
        self.root.withdraw()


# --- Screenshot Handling ---
def capture_frame() -> bytes:
    """Request a screenshot from `screenshot_capture_server.py`."""
    resp = requests.get(CAPTURE_SERVER_URL, timeout=5)
    resp.raise_for_status()
    return resp.content


def send_frame(img_bytes: bytes):
    files = {"screenshot": ("screenshot.png", io.BytesIO(img_bytes), "image/png")}
    headers = {"X-API-Key": API_KEY}
    try:
        resp = requests.post(SERVER_URL, files=files, headers=headers, timeout=2)
        resp.raise_for_status()
        data = resp.json()
        suggestion = data.get("suggestion", "No suggestion")
        return suggestion, data
    except Exception as e:
        return f"Error: {e}", None


# --- Main Loop ---
def main_loop(overlay: OverlayWindow):
    last_suggestion = "Connecting to server..."
    suggestion_lock = threading.Lock()

    def network_worker():
        nonlocal last_suggestion
        while True:
            start = time.time()
            img_bytes = capture_frame()
            suggestion, _ = send_frame(img_bytes)
            with suggestion_lock:
                last_suggestion = suggestion
            if ENABLE_TTS:
                speak(suggestion)
            elapsed = time.time() - start
            delay = FRAME_INTERVAL - elapsed
            if delay > 0:
                time.sleep(delay)

    net_thread = threading.Thread(target=network_worker, daemon=True)
    net_thread.start()

    while True:
        with suggestion_lock:
            overlay.set_text(last_suggestion)
        time.sleep(GUI_UPDATE_INTERVAL)


# --- Cleanup ---
def cleanup():
    global speech_running, speech_queue
    speech_running = False
    if speech_queue:
        speech_queue.put(None)
    if speech_thread and speech_thread.is_alive():
        speech_thread.join(timeout=0.5)


if __name__ == "__main__":
    try:
        overlay = OverlayWindow()
        threading.Thread(target=main_loop, args=(overlay,), daemon=True).start()
        overlay.show()
        overlay.root.mainloop()
    finally:
        cleanup()
