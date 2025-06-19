"""
hero_turn_detector.py – full revised script (rev-B)
===================================================
Determines if it is Hero’s turn to act by combining button-flash,
time-bar, card presence, and OCR cues.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Any, Dict

# ──────────────────────────────────────────────────────
#  Tunable constants
# ──────────────────────────────────────────────────────
BTN_HSV  = ((25, 100, 200), (80, 255, 255))    # highlight on action buttons
BAR_HSV  = ((20, 120, 120), (150, 255, 255))   # timer bar fill
BTN_THRESH      = 0.08
BAR_THRESH      = 0.25
ACTION_WHITE_TH = 0.05
CANNY_EDGES_TH  = 120
TEXT_CUES = ("YOUR TURN", "TO ACT", "TIME TO ACT", "ACTION REQUIRED")

try:
    import pytesseract
except ImportError:  # OCR optional
    pytesseract = None


# ─────────────────────────────────────────────────

def is_heros_turn(frame: np.ndarray, layout: Any, last_state: Dict) -> bool:
    """Return True if multiple cues agree it’s hero’s turn."""
    if detect_button_flash(frame, layout):
        return True
    if detect_time_bar(frame, layout):
        return True
    if hero_cards_present(last_state) and action_box_visible(frame, layout):
        return True
    if ocr_text := ocr_action_text(frame, layout):
        if any(cue in ocr_text.upper() for cue in TEXT_CUES):
            return True
    return False


# ─────────────────────────────────────────────────
#  Cue 1 – Button flash
# ─────────────────────────────────────────────────

def detect_button_flash(frame: np.ndarray, layout: Any) -> bool:
    roi = _crop_from_layout(frame, layout, "action_buttons")
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BTN_HSV[0], BTN_HSV[1])
    ratio = np.count_nonzero(mask) / mask.size
    return ratio > BTN_THRESH


# ─────────────────────────────────────────────────
#  Cue 2 – Time bar
# ─────────────────────────────────────────────────

def detect_time_bar(frame: np.ndarray, layout: Any) -> bool:
    roi = _crop_from_layout(frame, layout, "time_bar")
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BAR_HSV[0], BAR_HSV[1])
    ratio = np.count_nonzero(mask) / mask.size
    return ratio > BAR_THRESH


# ─────────────────────────────────────────────────
#  Cue 3 – Action box visibility
# ─────────────────────────────────────────────────

def action_box_visible(frame: np.ndarray, layout: Any) -> bool:
    roi = _crop_from_layout(frame, layout, "action_box")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # white ratio
    _, thresh = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
    white_ratio = np.count_nonzero(thresh) / thresh.size
    # edge contrast
    edges = cv2.Canny(gray, CANNY_EDGES_TH, CANNY_EDGES_TH * 2)
    edge_ratio = np.count_nonzero(edges) / edges.size
    return white_ratio > ACTION_WHITE_TH or edge_ratio > 0.03


# ─────────────────────────────────────────────────
#  Cue 4 – OCR prompt
# ─────────────────────────────────────────────────

def ocr_action_text(frame: np.ndarray, layout: Any) -> str:
    if pytesseract is None:
        return ""
    roi = _crop_from_layout(frame, layout, "action_prompt")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    txt = pytesseract.image_to_string(gray, config="--psm 7")
    return txt.strip()


# ─────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────

def hero_cards_present(last_state: Dict) -> bool:
    return bool(last_state.get("hole_cards")) and not last_state.get("hero_folded", False)


def _crop_from_layout(frame: np.ndarray, layout: Any, key: str) -> np.ndarray:
    """Supports both absolute pixel tuples & percent tuples."""
    try:
        x1, y1, x2, y2 = layout.get_zone_crop(key)  # PokerTableLayout API
        return frame[y1:y2, x1:x2]
    except Exception:
        # fallback: attribute or dict with (x,y,w,h) in px or %
        region = getattr(layout, key, None) or layout.get(key)
        return crop_region(frame, region)


def crop_region(frame: np.ndarray, region: Tuple[float, float, float, float]) -> np.ndarray:
    """Accepts (x, y, w, h) in px or (x%, y%, w%, h%)."""
    h, w = frame.shape[:2]
    x, y, rw, rh = region
    if 0 < x < 1 and 0 < y < 1:
        x, y, rw, rh = int(x * w), int(y * h), int(rw * w), int(rh * h)
    return frame[y : y + rh, x : x + rw]
