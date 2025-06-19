# realtime_pipeline.py  \u2013  multithreaded capture \u2192 OCR \u2192 decision \u2192 HUD
# ====================================================================
# \u2022 Clean shutdown with join-and-drain
# \u2022 Back-pressure: non-blocking put; counts dropped frames
# \u2022 Centralised exception logging
# \u2022 Adjustable FPS via constants
# \u2022 Placeholder stubs ready for real implementations

from __future__ import annotations
import logging
import queue
import threading
import time
from typing import Any, Tuple

# ────────────────────────────────────────
#  Config
# ────────────────────────────────────────
CAPTURE_FPS      = 30
OCR_WORKERS      = 1
QUEUE_MAXLEN     = 8
LOG_DROP_EVERY   = 1.0            # seconds
SHUTDOWN_TIMEOUT = 2.0

# ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(threadName)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger("pipeline")

# ────────────────────────────────────────
frame_q: queue.Queue[Any]              = queue.Queue(maxsize=QUEUE_MAXLEN)
state_q: queue.Queue[dict]             = queue.Queue(maxsize=QUEUE_MAXLEN)
dec_q:   queue.Queue[Tuple[dict, str]] = queue.Queue(maxsize=QUEUE_MAXLEN)
shutdown_event = threading.Event()

# ────────────────────────────────────────
#  Workers
# ────────────────────────────────────────
def screen_capture_loop() -> None:
    frame_interval = 1.0 / CAPTURE_FPS
    dropped_frames = 0
    last_report    = time.time()
    frame_no       = 0

    while not shutdown_event.is_set():
        t0 = time.perf_counter()
        try:
            frame_no += 1
            frame = capture_frame(frame_no)            # ← your real capture
            try:
                frame_q.put_nowait(frame)
            except queue.Full:
                dropped_frames += 1
        except Exception:
            log.exception("capture error")

        # periodic drop count
        if time.time() - last_report >= LOG_DROP_EVERY:
            if dropped_frames:
                log.warning("Dropped frames: %d", dropped_frames)
                dropped_frames = 0
            last_report = time.time()

        dt = time.perf_counter() - t0
        if dt < frame_interval:
            time.sleep(frame_interval - dt)


def ocr_loop(worker_id: int) -> None:
    while not shutdown_event.is_set():
        try:
            frame = frame_q.get(timeout=0.5)
            state = parse_ocr_and_cards(frame)         # ← your OCR
            frame_q.task_done()
            try:
                state_q.put_nowait(state)
            except queue.Full:
                log.debug("State queue full — dropping state")
        except queue.Empty:
            continue
        except Exception:
            log.exception("OCR error")


def decision_loop() -> None:
    while not shutdown_event.is_set():
        try:
            gs = state_q.get(timeout=0.5)
            rec = decision_engine(gs)                  # ← your decision logic
            state_q.task_done()
            try:
                dec_q.put_nowait((gs, rec))
            except queue.Full:
                log.debug("Decision queue full — dropping recommendation")
        except queue.Empty:
            continue
        except Exception:
            log.exception("Decision error")


def ui_loop() -> None:
    while not shutdown_event.is_set():
        try:
            gs, rec = dec_q.get(timeout=0.5)
            update_overlay(gs, rec)                    # ← your HUD
            play_audio_alert(rec)                      # ← your audio
            dec_q.task_done()
        except queue.Empty:
            continue
        except Exception:
            log.exception("UI error")

# ────────────────────────────────────────
#  Placeholders (replace with production code)
# ────────────────────────────────────────
def capture_frame(num: int) -> Any:
    time.sleep(0.03)
    return f"frame-{num}"

def parse_ocr_and_cards(frame: Any) -> dict:
    time.sleep(0.15)
    return {"game_state": frame}

def decision_engine(gs: dict) -> str:
    time.sleep(0.02)
    return "RECOMMEND: CALL"

def update_overlay(gs: dict, rec: str) -> None:
    log.info(f"HUD: {rec}")

def play_audio_alert(rec: str) -> None:
    log.info(f"Speaker: {rec}")

# ────────────────────────────────────────
#  Launch threads
# ────────────────────────────────────────
threads: list[threading.Thread] = [
    threading.Thread(target=screen_capture_loop, name="CaptureWorker"),
    threading.Thread(target=decision_loop,      name="DecisionWorker"),
    threading.Thread(target=ui_loop,            name="UIWorker"),
]

for i in range(OCR_WORKERS):
    threads.append(threading.Thread(target=ocr_loop, name=f"OCRWorker-{i}", args=(i,)))

for t in threads:
    t.start()

# ────────────────────────────────────────
#  Main loop / graceful shutdown
# ────────────────────────────────────────
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    log.info("Shutting down …")
    shutdown_event.set()
    for q in (frame_q, state_q, dec_q):
        try:
            q.join()
        except AttributeError:
            pass
    for t in threads:
        t.join(timeout=SHUTDOWN_TIMEOUT)
    log.info("Bye!")
