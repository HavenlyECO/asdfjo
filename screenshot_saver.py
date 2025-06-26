"""Capture screenshots and send them to the server."""

import io
import time
import requests
import mss

from local_client import capture_frame, TARGET_FPS


SERVER_URL = "http://localhost:5001/api/save_screenshot"
FRAME_INTERVAL = 1.0 / TARGET_FPS


def send_frame(img):
    """Upload a screenshot image to the server."""
    img_bytes = mss.tools.to_png(img.rgb, img.size)
    files = {"screenshot": ("screenshot.png", io.BytesIO(img_bytes), "image/png")}
    resp = requests.post(SERVER_URL, files=files, timeout=5)
    resp.raise_for_status()
    return resp.json()


def main():
    while True:
        start = time.time()
        img = capture_frame()
        result = send_frame(img)
        print(result)
        elapsed = time.time() - start
        delay = FRAME_INTERVAL - elapsed
        if delay > 0:
            time.sleep(delay)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
