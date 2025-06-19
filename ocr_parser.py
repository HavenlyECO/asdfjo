"""
OCRParser – **adaptive blind‑box edition**  (2025‑06‑18 rev‑C)
==============================================================

This version starts from the code you pasted and layers in three adaptive
features:

1. **Dynamic blind‑box discovery** – if the layout doesn’t define `blind_box`,
   the parser scans the top 15% of the screenshot once per level, finds the
   text that matches `/\d+[kK]?\s*/\s*\d+[kK]?/`, grabs its bounding box, and
   inserts that box into `layout.zones` at runtime.  It then caches it so every
   subsequent frame hits a fixed ROI (keeps CPU low).
2. **Toggle between Otsu ↔ adaptive threshold** via `bin_mode` param
   (`"otsu"` by default).
3. **Digits‑only Tesseract mode** for numeric fields (stack, pot) to reduce
   mis‑reads.

Everything else (EasyOCR fallback, debug overlay) sticks to your original
interfaces.
"""
from __future__ import annotations

import re
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np


class OCRParser:
    """Extract poker table state from a screenshot using adaptive ROI for blinds."""

    # ------------------------------------------------------------------
    #   Init
    # ------------------------------------------------------------------
    def __init__(self, *,
                 use_easyocr: bool = True,
                 bin_mode: str = "otsu",            # "otsu" | "adaptive"
                 easyocr_langs: list[str] | None = None):
        self.use_easyocr = use_easyocr
        self.bin_mode    = bin_mode
        self.cached_blind_box: Optional[Tuple[int,int,int,int]] = None  # (x1,y1,x2,y2)

        if use_easyocr:
            try:
                import easyocr
            except ModuleNotFoundError:
                raise RuntimeError("EasyOCR not installed – pip install easyocr or set use_easyocr=False")
            langs = easyocr_langs or ["en"]
            self.reader = easyocr.Reader(langs, gpu=False)
        else:
            import pytesseract
            self.reader = pytesseract

    # ------------------------------------------------------------------
    #   Pre-processing helpers
    # ------------------------------------------------------------------
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        if self.bin_mode == "adaptive":
            return cv2.adaptiveThreshold(upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 21, 8)
        # Otsu default
        blur  = cv2.GaussianBlur(upscaled, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    # ------------------------------------------------------------------
    #   OCR dispatch
    # ------------------------------------------------------------------
    def _ocr(self, img: np.ndarray, *, digits_only=False) -> str:
        if self.use_easyocr:
            out = self.reader.readtext(img, detail=0)
            return " ".join(out).strip() if out else ""
        # pytesseract
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cfg = "--psm 7"
        if digits_only:
            cfg += " -c tessedit_char_whitelist=0123456789."
        return self.reader.image_to_string(rgb, config=cfg).strip()

    # ------------------------------------------------------------------
    #   Text cleanup helpers
    # ------------------------------------------------------------------
    _NUM_RE  = re.compile(r"(\d+\.?\d*)")
    _BLIND_RE= re.compile(r"\d+[kK]?\s*/\s*\d+[kK]?")

    @classmethod
    def _to_float(cls, txt: str) -> Optional[float]:
        txt = txt.replace("O", "0").replace("l", "1").replace("I", "1")
        m = cls._NUM_RE.search(txt.replace(",", ""))
        return float(m.group()) if m else None

    @classmethod
    def _clean_blind(cls, txt: str) -> Optional[str]:
        txt = txt.replace("O", "0").replace("l", "1").replace("I", "1")
        txt = txt.replace(" ", "")
        return txt if "/" in txt else None

    # ------------------------------------------------------------------
    #   Adaptive blind-box finder
    # ------------------------------------------------------------------
    def _find_blind_box(self, img: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        """Scan the top 15% strip for a blind-level pattern; return bounding box or None."""
        h, w, _ = img.shape
        search_area = img[0:int(0.15*h), :]
        if self.use_easyocr:
            results = self.reader.readtext(self._preprocess(search_area), detail=1)
            for bbox, text, _ in results:
                if self._BLIND_RE.fullmatch(text.replace(" ", "")):
                    # bbox is [[x1,y1], [x2,y2], ...]; take extremes
                    xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
                    return (min(xs), min(ys), max(xs), max(ys))
        else:
            data = self.reader.image_to_data(self._preprocess(search_area), config="--psm 6", output_type=self.reader.Output.DICT)
            for i, word in enumerate(data["text"]):
                if self._BLIND_RE.fullmatch(word.replace(" ", "")):
                    x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    return (x, y, x+bw, y+bh)
        return None

    # ------------------------------------------------------------------
    #   Public parse()
    # ------------------------------------------------------------------
    def parse(self, screenshot: np.ndarray, layout) -> Dict[str, Any]:
        zones: Dict[str, str] = {
            "hero_stack":  "hero_stack",
            "pot":         "pot_area",
            "action_text": "action_box",
        }

        # Ensure blind_box exists – discover if missing
        if "blind_box" not in layout.get_all_zones():
            if self.cached_blind_box is None:
                bb = self._find_blind_box(screenshot)
                if bb:
                    self.cached_blind_box = bb
            if self.cached_blind_box:
                layout.zones["blind_box"] = self.cached_blind_box
        if "blind_box" in layout.get_all_zones():
            zones["blind_level"] = "blind_box"

        for z in layout.get_all_zones():
            if z.startswith("opponent_") and z.endswith("_stack"):
                zones[z] = z

        state: Dict[str, Any] = {"opponents": {}}

        for logical_key, roi in zones.items():
            try:
                x1,y1,x2,y2 = layout.get_zone_crop(roi)
                if x2<=x1 or y2<=y1:
                    continue
                crop = screenshot[y1:y2, x1:x2]
                proc = self._preprocess(crop)
                txt  = self._ocr(proc, digits_only=logical_key in {"hero_stack", "pot"})

                if logical_key == "hero_stack":
                    state["hero_stack"] = self._to_float(txt)
                elif logical_key == "pot":
                    state["pot"] = self._to_float(txt)
                elif logical_key == "blind_level":
                    state["blind_level"] = self._clean_blind(txt)
                elif logical_key == "action_text":
                    state["action_text"] = txt
                elif logical_key.startswith("opponent_"):
                    val = self._to_float(txt)
                    if val is not None:
                        seat = logical_key.replace("_stack", "")
                        state["opponents"][seat] = val
            except Exception as err:
                print(f"[OCRParser] {logical_key}: {err}")
                continue
        return state

    # ------------------------------------------------------------------
    #   Debug overlay
    # ------------------------------------------------------------------
    @staticmethod
    def draw_layout(img: np.ndarray, layout) -> np.ndarray:
        vis = img.copy()
        for name, (x1,y1,x2,y2) in layout.get_all_zones().items():
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, name, (x1, max(y1-8, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,255), 1)
        return vis


# ----------------------------------------------------------------------
#   Quick harness (remove in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from poker_table_layout_acr import PokerTableLayout
    img = cv2.imread("/mnt/data/Screenshot 2025-06-18 180416.png")
    table = PokerTableLayout(img.shape[1], img.shape[0], "mtt")
    parser= OCRParser(use_easyocr=False)
    print(parser.parse(img, table))
    cv2.imwrite("/mnt/data/overlay_revC.png", OCRParser.draw_layout(img, table))
    print("overlay_revC.png written")
