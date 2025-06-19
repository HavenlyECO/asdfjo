# realtime_poker_pipeline.py  –  end-to-end capture \u2192 OCR \u2192 solve \u2192 HUD
# ====================================================================

from __future__ import annotations
import concurrent.futures
import logging
import queue
import threading
import time
from collections import deque
from typing import Any, Tuple

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────
#  Place-holder imports (replace with real libs)
# ──────────────────────────────────────────────────────────────
def capture_screen():
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)  # demo frame

def get_crops(frame):
    return {
        "pot": frame[0:80, 600:680],
        "hero_cards": [frame[600:720, 400:460], frame[600:720, 460:520]],
        "board_cards": [],
        "villain_cards": [],
        "action_box": frame[630:700, 540:740],
    }

def batch_ocr(imgs):   return [("1000", 0.9) for _ in imgs]
def batch_card_detect(imgs): return [("AS", 0.9) for _ in imgs]
def parse_decision(gs): return "RECOMMEND: CALL"

# ──────────────────────────────────────────────────────────────
#  Frame-change detector (dHash + edge diff)
# ──────────────────────────────────────────────────────────────
from frame_change_detector import FrameChangeDetector
_detector = FrameChangeDetector(threshold=0.03)

# ──────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(threadName)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger("pipeline")

# ──────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────
CAPTURE_FPS      = 30
QUEUE_MAXLEN     = 4
SHUTDOWN         = threading.Event()

# ──────────────────────────────────────────────────────────────
#  Queues
# ──────────────────────────────────────────────────────────────
frame_q  : queue.Queue[np.ndarray]         = queue.Queue(maxsize=QUEUE_MAXLEN)
result_q : queue.Queue[Tuple[dict, str]]   = queue.Queue(maxsize=QUEUE_MAXLEN)

# ──────────────────────────────────────────────────────────────
#  FPS monitor
# ──────────────────────────────────────────────────────────────
class FPSMonitor:
    def __init__(self, window=30, target=10):
        self.win = deque(maxlen=window)
        self.target = target

    def tick(self, dt: float):
        self.win.append(dt)
        if len(self.win) == self.win.maxlen:
            fps = 1 / (sum(self.win) / len(self.win))
            if fps < self.target:
                log.warning("LOW pipeline FPS %.1f (target %d)", fps, self.target)
            else:
                log.info("Pipeline FPS %.1f", fps)

fps_monitor = FPSMonitor(window=30, target=12)

# ──────────────────────────────────────────────────────────────
#  Worker thread
# ──────────────────────────────────────────────────────────────
class PokerPipelineWorker(threading.Thread):
    def __init__(self):
        super().__init__(name="PipelineWorker")
        self.last_game_state: dict | None = None
        self.last_recommendation: str | None = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def run(self):
        while not SHUTDOWN.is_set():
            try:
                frame = frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            t0 = time.perf_counter()

            # Frame diff
            if (self.last_game_state is not None
                    and not _detector.has_significant_change("table", frame)):
                result_q.put_nowait((self.last_game_state, self.last_recommendation))
                continue

            try:
                crops = get_crops(frame)
            except Exception as err:
                log.exception(err)
                continue

            ocr_imgs   = [crops["pot"], crops["action_box"]]
            card_imgs  = crops["hero_cards"] + crops["board_cards"] + crops["villain_cards"]

            # parallel OCR + cards
            fut_ocr  = self.executor.submit(batch_ocr, ocr_imgs)
            fut_card = self.executor.submit(batch_card_detect, card_imgs)

            ocr_res, card_res = fut_ocr.result(), fut_card.result()

            gs = build_game_state_from_results(ocr_res, card_res, crops)
            rec = parse_decision(gs)

            self.last_game_state = gs
            self.last_recommendation = rec
            result_q.put_nowait((gs, rec))

            fps_monitor.tick(time.perf_counter() - t0)

# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────
def build_game_state_from_results(ocr_results, card_results, crops):
    return {
        "pot": ocr_results[0][0] if ocr_results else None,
        "hero_card_labels": [c[0] for c in card_results[:2]] if card_results else None,
    }

# ──────────────────────────────────────────────────────────────
#  Capture loop
# ──────────────────────────────────────────────────────────────
def capture_loop():
    frame_interval = 1.0 / CAPTURE_FPS
    while not SHUTDOWN.is_set():
        t0 = time.perf_counter()
        try:
            frame = capture_screen()
            if frame is None:
                time.sleep(0.1)
                continue
            try:
                frame_q.put_nowait(frame)
            except queue.Full:
                pass  # drop frame
        except Exception as err:
            log.exception(err)
        dt = time.perf_counter() - t0
        sleep = frame_interval - dt
        if sleep > 0:
            time.sleep(sleep)

# ──────────────────────────────────────────────────────────────
#  Main loop
# ──────────────────────────────────────────────────────────────
def main():
    worker = PokerPipelineWorker()
    worker.start()

    cap_thread = threading.Thread(target=capture_loop, name="CaptureThread")
    cap_thread.start()

    try:
        while True:
            gs, rec = result_q.get(timeout=1)
            log.info("Recommendation: %s  |  Pot=%s", rec, gs.get("pot"))
    except KeyboardInterrupt:
        log.info("Shutting down …")
        SHUTDOWN.set()
        worker.join(3)
        cap_thread.join(3)

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
