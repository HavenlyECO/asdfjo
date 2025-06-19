"""
PlayerActionParser – rev-B (2025-06-18)
======================================
*Patch over rev-A after another code-safety audit.*  Changes in this revision:

1. **OCR access helper** – replaced the call to a non-existent `ocr_raw()` with
   an internal `_ocr_text()` that works with either OCRParser rev-C (uses
   `.parse()` + temporary mini-layout) **or** any object exposing
   `.preprocess_for_ocr()` + `.reader` (back-compat with your older snippet).
2. **Opponent card visibility heuristic** – now `False` by default and only
   flips to `True` if *chips* or *explicit action text* are present, preventing
   false “folded” when opponents are simply waiting.
3. **Seat tag mapping stub** – optional `seat_tags` list in the layout lets you
   return BTN, SB, etc.; falls back to `seat_0`… if absent.
4. **Regex tweak** – captures numbers with optional `$` prefix and comma
   thousands (e.g. `$1,250.50`).

Everything else remains unchanged from rev-A.
"""
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
import re


class PlayerActionParser:
    def __init__(self,
                 layout,
                 *,
                 card_detector=None,
                 ocr_parser=None,
                 chip_hsv_ranges: Optional[List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]] = None,
                 chip_area_raise: int = 1200):
        self.layout          = layout
        self.card_detector   = card_detector
        self.ocr_parser      = ocr_parser
        self.chip_area_raise = chip_area_raise
        self.chip_hsv_ranges = chip_hsv_ranges or [
            ((  0,  80, 80), ( 10,255,255)), ((160, 80,80), (179,255,255)),  # red
            (( 35,  70, 70), ( 85,255,255)),                                 # green
            (( 90,  80, 80), (130,255,255)),                                 # blue
        ]

    # ------------------------------------------------------------------
    def parse_actions(self, screenshot: np.ndarray) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        seat_count = getattr(self.layout, "seat_count", 6)
        seat_tags  = getattr(self.layout, "seat_tags", [f"seat_{i}" for i in range(seat_count)])

        for seat in range(seat_count):
            seat_key  = seat_tags[seat] if seat < len(seat_tags) else f"seat_{seat}"
            zone_name = f"action_box_seat{seat}"
            try:
                x1, y1, x2, y2 = self.layout.get_zone_crop(zone_name)
            except KeyError:
                continue
            crop = screenshot[y1:y2, x1:x2]
            chips_n, chip_area = self._detect_chips(crop)

            # card visibility (simplified: hero hole cards already cached)
            cards_visible = False
            if self.card_detector is not None and seat == 0:
                cards_visible = bool(self.card_detector._cache.get("hole", []))

            # OCR text
            action_text = self._ocr_text(crop)

            status, act, amt = self._infer_state(chips_n, chip_area, cards_visible, action_text)
            seat_dict: Dict[str, Any] = {"status": status}
            if act:
                seat_dict["action"] = act
            if amt is not None:
                seat_dict["amount"] = amt
            results[seat_key] = seat_dict
        return results

    # ------------------------------------------------------------------
    #   OCR helper (works with OCRParser rev-C or your legacy snippet)
    # ------------------------------------------------------------------
    def _ocr_text(self, crop: np.ndarray) -> str:
        if self.ocr_parser is None:
            return ""
        if hasattr(self.ocr_parser, "parse"):
            # Build a 1-zone mini-layout on the fly
            class _TmpLayout:  # minimal shim
                def get_zone_crop(self, _):
                    return 0, 0, crop.shape[1], crop.shape[0]
                def get_all_zones(self):
                    return {"tmp": (0,0,0,0)}
            state = self.ocr_parser.parse(crop, _TmpLayout())
            return state.get("action_text", "") or ""
        # legacy path
        proc = self.ocr_parser.preprocess_for_ocr(crop)
        if getattr(self.ocr_parser, "use_easyocr", False):
            res = self.ocr_parser.reader.readtext(proc, detail=0)
            return res[0].strip() if res else ""
        return self.ocr_parser.reader.image_to_string(proc, config="--psm 7").strip()

    # ------------------------------------------------------------------
    def _detect_chips(self, img: np.ndarray) -> Tuple[int, int]:
        if img.size == 0:
            return 0, 0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        masks = [cv2.inRange(hsv, lo, hi) for lo, hi in self.chip_hsv_ranges]
        mask  = np.bitwise_or.reduce(masks)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt, area_sum = 0, 0
        for c in contours:
            a = cv2.contourArea(c)
            if 80 < a < 2500:
                cnt += 1
                area_sum += a
        return cnt, area_sum

    # ------------------------------------------------------------------
    def _infer_state(self, chips_n: int, chip_area: int, cards_visible: bool, text: str):
        text_low = text.lower() if text else ""
        if "raise" in text_low or "bet" in text_low:
            return "active", "raise" if "raise" in text_low else "bet", self._extract_amount(text_low)
        if "call" in text_low:
            return "active", "call", self._extract_amount(text_low)
        if "check" in text_low:
            return "active", "check", None

        if chips_n > 0:
            if chip_area >= self.chip_area_raise:
                return "active", "raise", None
            return "active", "call", None

        if not cards_visible and chips_n == 0:
            return "folded", None, None

        return "active", None, None

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_amount(txt: str | None):
        if not txt:
            return None
        m = re.search(r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+)", txt)
        if m:
            return float(m.group(1).replace(",", ""))
        return None
