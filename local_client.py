"""
Poker Assistant Local Thin Client
---------------------------------

Requirements:
- Python 3.8+
- mss
- requests
- tkinter
- pyttsx3 (optional, for TTS)
- (optional: keyboard or pynput for hotkeys)

Setup Instructions:
1. pip install -r requirements.txt
2. Edit the config section for your server URL, API key, and the screen region to capture.
3. Run: python local_client.py

Notes:
- No business logic is done locally; this only captures the screen, sends to server, and displays advice.
- Does NOT interact directly with any poker client (just captures image region).

Author: <Your Name>
"""

import io
import time
import threading
import requests
import mss
from tkinter import Tk, Label, StringVar
import pyttsx3

# --- CONFIGURATION ---
SERVER_URL = "http://YOUR_SERVER_IP:5000/api/advice"  # Set your server address here
API_KEY = "changeme"  # Set to match your server's API key
CAPTURE_REGION = {"top": 100, "left": 100, "width": 800, "height": 600}  # Adjust to table location
LOOP_INTERVAL = 3  # seconds
ENABLE_TTS = True  # Set False to disable text-to-speech

# --- Overlay UI ---
class OverlayWindow:
    def __init__(self):
        self.root = Tk()
        self.root.overrideredirect(True)  # Remove border
        self.root.attributes('-topmost', True)
        self.root.geometry("+100+100")
        self.var = StringVar()
        self.label = Label(self.root, textvariable=self.var, font=("Arial", 28), bg="yellow", fg="black", padx=10, pady=10)
        self.label.pack()
        self.var.set("Waiting for advice...")

    def set_text(self, txt):
        self.var.set(txt)
        self.root.update()

    def show(self):
        self.root.deiconify()
        self.root.update()

    def hide(self):
        self.root.withdraw()

# --- TTS Helper ---

def speak(text):
    if not ENABLE_TTS:
        return
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# --- Screenshot and Network ---

def capture_and_send():
    with mss.mss() as sct:
        img = sct.grab(CAPTURE_REGION)
        img_bytes = mss.tools.to_png(img.rgb, img.size)
        files = {'screenshot': ("screenshot.png", io.BytesIO(img_bytes), "image/png")}
        headers = {"X-API-Key": API_KEY}
        try:
            resp = requests.post(SERVER_URL, files=files, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            suggestion = data.get("suggestion", "No suggestion")
            return suggestion, data
        except Exception as e:
            return f"Error: {e}", None

# --- Main Loop ---

def main_loop(overlay):
    while True:
        suggestion, data = capture_and_send()
        overlay.set_text(suggestion)
        if ENABLE_TTS:
            threading.Thread(target=speak, args=(suggestion,), daemon=True).start()
        time.sleep(LOOP_INTERVAL)

if __name__ == "__main__":
    overlay = OverlayWindow()
    threading.Thread(target=main_loop, args=(overlay,), daemon=True).start()
    overlay.show()
    overlay.root.mainloop()
