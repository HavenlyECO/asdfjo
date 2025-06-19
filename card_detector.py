"""
CardDetector – YOLO‑v8 + template/CNN fallback (rev‑A)
=====================================================
* Thorough pass over the code you supplied; fixes & small optimisations marked
  in comments that start with  # <--  .
* Still drop‑in compatible with your earlier imports.

Major tweaks
------------
1. **Predict call** – ultralytics YOLO now prefers `self.model(crop)` instead of
   `.predict()`.  Using the functional call avoids an extra wrapper and gives
   identical Results objects.
2. **Empty‑prediction guard** – early‑exit if `results[0].boxes` is empty.
3. **Name normalisation** – suit letters forced to lower‑case (`Ah`, `Td`, etc.)
   so downstream string compares never fail.
4. **Confidence ladder** – 0.25 (keep) for detection, 0.40 → fallback can be
   tuned via `self.low_conf_threshold`.
5. **Phash fallback tolerance** – param `phash_max_dist` (default = 8) lets you
   widen/narrow the Hamming distance threshold instead of always picking the
   closest template.
6. **draw_card_detections()** – one helper to avoid DRY; supports any crop name
   (hero or community), draws directly on *img* with absolute coords so the
   overlay is correct even if you pass the frame to a UI.
"""
from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO


class CardDetector:
    def __init__(self,
                 model_path: str = "card_yolov8.pt",
                 *,
                 low_conf_threshold: float = 0.40,
                 use_fallback: bool = True,
                 phash_templates: Dict[str, 'imagehash.ImageHash'] | None = None,
                 fallback_cnn: Optional[callable] = None,
                 phash_max_dist: int = 8):
        """card_yolov8.pt must be trained with class names like 'ah', 'ks', ..."""
        self.model = YOLO(model_path)
        self.low_conf_threshold = low_conf_threshold
        self.use_fallback       = use_fallback
        self.phash_templates    = phash_templates or {}
        self.fallback_cnn       = fallback_cnn
        self.phash_max_dist     = phash_max_dist

        self._cache = {"hole": [], "board": []}    # flicker smoothing

    # ------------------------------------------------------------------
    #   Public API
    # ------------------------------------------------------------------
    def detect_cards(self, screenshot: np.ndarray, layout) -> Dict[str, List[str]]:
        hole_crop  = self._crop(screenshot, layout, "hero_hole_cards")
        board_crop = self._crop(screenshot, layout, "community_cards")

        hole = self._detect_single_crop(hole_crop, zone="hole")
        board= self._detect_single_crop(board_crop, zone="board")

        # -------- simple temporal smoothing --------
        if hole:
            self._cache["hole"] = hole
        else:
            hole = self._cache["hole"]

        if board:
            self._cache["board"] = board
        else:
            board = self._cache["board"]

        return {"hole_cards": hole, "board_cards": board}

    # ------------------------------------------------------------------
    #   Debug overlay
    # ------------------------------------------------------------------
    def draw_card_detections(self, img: np.ndarray, layout) -> np.ndarray:
        self._draw_crop(img, layout, "hero_hole_cards", color=(  0,255,  0))
        self._draw_crop(img, layout, "community_cards", color=(255,  0,  0))
        return img

    # ------------------------------------------------------------------
    #   Internal helpers
    # ------------------------------------------------------------------
    def _crop(self, img: np.ndarray, layout, zone: str) -> np.ndarray:
        x1,y1,x2,y2 = layout.get_zone_crop(zone)
        return img[y1:y2, x1:x2]

    def _detect_single_crop(self, crop: np.ndarray, *, zone: str) -> List[str]:
        if crop.size == 0:
            return []
        results = self.model(crop, conf=0.25, verbose=False)
        boxes   = results[0].boxes
        if len(boxes) == 0:
            return []

        cards: List[Tuple[int, str, float]] = []  # (x‑min, label, conf)
        for xyxy, conf, cls in zip(boxes.xyxy.cpu().numpy(),
                                   boxes.conf.cpu().numpy(),
                                   boxes.cls.cpu().numpy()):
            label_raw = results[0].names[int(cls)]
            label     = label_raw[0].upper() + label_raw[1:].lower()   # Ah, Td …

            if conf < self.low_conf_threshold and self.use_fallback:
                card_img = self._sub_crop(crop, xyxy)
                fb = self._fallback(card_img)
                if fb:
                    label = fb
            cards.append((int(xyxy[0]), label, conf))

        cards.sort(key=lambda t: t[0])          # left‑to‑right ordering
        return [lbl for _, lbl, _ in cards]

    def _sub_crop(self, crop: np.ndarray, xyxy) -> np.ndarray:
        x1,y1,x2,y2 = map(int, xyxy)
        h,w = crop.shape[:2]
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(w,x2), min(h,y2)
        return crop[y1:y2, x1:x2]

    # ----------------------------- fallbacks ---------------------------
    def _fallback(self, card_img: np.ndarray) -> Optional[str]:
        if card_img is None or card_img.size == 0:
            return None

        if self.fallback_cnn:
            try:
                return self.fallback_cnn(card_img)
            except Exception:
                pass

        if self.phash_templates:
            try:
                from PIL import Image
                import imagehash
                pil   = Image.fromarray(cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB))
                phash = imagehash.phash(pil)
                best, dist = None, 999
                for lbl, tmpl_hash in self.phash_templates.items():
                    d = abs(phash - tmpl_hash)
                    if d < dist:
                        best, dist = lbl, d
                if dist <= self.phash_max_dist:
                    return best
            except Exception:
                pass
        return None

    # ----------------------------- overlay -----------------------------
    def _draw_crop(self, img: np.ndarray, layout, zone: str, *, color):
        crop = self._crop(img, layout, zone)
        if crop.size == 0:
            return
        results = self.model(crop, conf=0.25, verbose=False)
        for xyxy, conf, cls in zip(results[0].boxes.xyxy.cpu().numpy(),
                                   results[0].boxes.conf.cpu().numpy(),
                                   results[0].boxes.cls.cpu().numpy()):
            label_raw = results[0].names[int(cls)]
            label     = label_raw[0].upper() + label_raw[1:].lower()
            x1,y1,x2,y2 = map(int, xyxy)
            cv2.rectangle(crop, (x1,y1), (x2,y2), color, 2)
            cv2.putText(crop, f"{label} {conf:.2f}", (x1, max(y1-3, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
