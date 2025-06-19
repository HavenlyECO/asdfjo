# visual_confidence.py  –  robust ensemble card/OCR detection (rev-B)
# ================================================================

from __future__ import annotations
import logging
import random
from collections import Counter
from typing import Dict, Tuple

# ───────────────────────────────────────────────────────
#  Logger
# ───────────────────────────────────────────────────────
logger = logging.getLogger("visual_confidence")
logger.setLevel(logging.INFO)
if not logger.handlers:
    hdlr = logging.StreamHandler()
    hdlr.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(hdlr)

# ───────────────────────────────────────────────────────
#  Card detection (YOLO + hash + CNN)
# ───────────────────────────────────────────────────────

def detect_card_region(
    img,
    *,
    min_confidence: float = 0.8,
    fallback_conf: float = 0.80,
) -> Dict[str, float | str]:
    """
    Returns {'label': str, 'confidence': float}.  If uncertain → 'UNKNOWN'.
    """
    y_label, y_conf = yolo_model(img)
    if y_conf >= min_confidence:
        return {"label": y_label, "confidence": y_conf}

    h_label = hash_match(img)
    c_label = cnn_classify(img)
    votes = [y_label, h_label, c_label]
    maj = _majority(votes)

    if maj != "UNKNOWN" and votes.count(maj) >= 2:
        logger.info(f"Ensemble override: {votes} → {maj}")
        return {"label": maj, "confidence": fallback_conf}

    logger.warning(f"Low-conf card parse: {votes} (YOLO={y_conf:.2f})")
    return {"label": "UNKNOWN", "confidence": 0.0}


def card_label_only(img) -> str:
    return detect_card_region(img)["label"]

# ───────────────────────────────────────────────────────
#  OCR ensemble
# ───────────────────────────────────────────────────────

def ocr_read(img, *, min_confidence: float = 0.7) -> str:
    t_txt, t_conf = tesseract_ocr(img)
    e_txt, e_conf = easyocr_ocr(img)
    t_conf = max(0.0, min(t_conf, 1.0))
    e_conf = max(0.0, min(e_conf, 1.0))

    if (t_conf >= min_confidence and e_conf >= min_confidence
            and t_txt.strip().lower() == e_txt.strip().lower()):
        return t_txt
    if t_conf >= min_confidence and t_conf >= e_conf:
        return t_txt
    if e_conf >= min_confidence and e_conf > t_conf:
        return e_txt

    logger.warning(
        f"OCR low-conf: Tesseract=({t_txt},{t_conf:.2f}) "
        f"EasyOCR=({e_txt},{e_conf:.2f})"
    )
    return "UNKNOWN"

# ───────────────────────────────────────────────────────
#  Game-state helper
# ───────────────────────────────────────────────────────

def update_game_state(gs: Dict, key: str, value):
    if value == "UNKNOWN" or gs.get(key) == value:
        return
    gs[key] = value

# ───────────────────────────────────────────────────────
#  Utilities
# ───────────────────────────────────────────────────────

def _majority(lst) -> str:
    filt = [x for x in lst if x != "UNKNOWN"]
    return Counter(filt).most_common(1)[0][0] if filt else "UNKNOWN"

# ───────────────────────────────────────────────────────
#  Placeholder stubs (replace with real models/engines)
# ───────────────────────────────────────────────────────

def yolo_model(img) -> Tuple[str, float]:
    return random.choice(["AS", "KD", "UNKNOWN"]), random.uniform(0.5, 0.95)

def hash_match(img) -> str:
    return random.choice(["AS", "KD", "UNKNOWN"])

def cnn_classify(img) -> str:
    return random.choice(["AS", "KD", "UNKNOWN"])

def tesseract_ocr(img) -> Tuple[str, float]:
    return "1500", random.uniform(0.5, 0.9)

def easyocr_ocr(img) -> Tuple[str, float]:
    return "1500", random.uniform(0.5, 0.9)

# ───────────────────────────────────────────────────────
#  Demo loop
# ───────────────────────────────────────────────────────
if __name__ == "__main__":
    dummy = object()
    gs = {}
    for _ in range(5):
        update_game_state(gs, "hero_card1", card_label_only(dummy))
        update_game_state(gs, "pot", ocr_read(dummy))
        print(gs)
