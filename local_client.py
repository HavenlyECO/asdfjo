"""
Poker Assistant Local Client
---------------------------------
"""

import io
import time
import threading
import queue
import requests
import mss
from tkinter import Tk, Label, StringVar
import pyttsx3

# --- CONFIGURATION ---
SERVER_URL = "http://24.199.98.206:5000/api/advice" 
API_KEY = "your-secure-api-key"  # Set to match your server's API key
CAPTURE_REGION = {"top": 230, "left": 280, "width": 960, "height": 720}  # ACR dimensions
LOOP_INTERVAL = 3  # seconds
ENABLE_TTS = True  # Set False to disable text-to-speech

# --- Speech Engine Management ---
speech_queue = queue.Queue()
speech_engine = None
speech_thread = None
speech_running = False

def speech_worker():
    """Dedicated thread for TTS to prevent concurrent engine access"""
    global speech_engine, speech_running
    
    speech_engine = pyttsx3.init()
    speech_running = True
    
    while speech_running:
        try:
            text = speech_queue.get(timeout=0.5)
            if text is None:  # Signal to stop the thread
                break
                
            speech_engine.say(text)
            speech_engine.runAndWait()
            speech_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"TTS Error: {e}")
            time.sleep(0.1)  # Prevent CPU thrashing on repeated errors

def speak(text):
    """Queue text for speaking without blocking"""
    if not ENABLE_TTS:
        return
    
    global speech_thread
    
    # Start speech thread if it doesn't exist
    if speech_thread is None or not speech_thread.is_alive():
        speech_thread = threading.Thread(target=speech_worker, daemon=True)
        speech_thread.start()
    
    # Add text to queue (non-blocking)
    speech_queue.put(text)

# --- Overlay UI ---
class OverlayWindow:
    def __init__(self):
        self.root = Tk()
        self.root.overrideredirect(True)  # Remove border
        self.root.attributes('-topmost', True)
        self.root.geometry("+100+100")
        self.var = StringVar()
        self.label = Label(self.root, textvariable=self.var, font=("Arial", 28), 
                          bg="yellow", fg="black", padx=10, pady=10)
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
            speak(suggestion)
            
        time.sleep(LOOP_INTERVAL)

# --- Cleanup Function ---
def cleanup():
    """Ensure proper cleanup of resources"""
    global speech_running, speech_queue
    
    speech_running = False
    if speech_queue:
        speech_queue.put(None)  # Signal to stop the thread
    
    # Wait briefly for the thread to clean up
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
