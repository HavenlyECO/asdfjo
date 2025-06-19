import cv2
import numpy as np
import hashlib

class OCRParser:
    """
    OCRParser with value caching, change detection, and fallback hooks.
    Fallback is dormant unless enabled and accuracy-validated.
    """

    def __init__(self, use_easyocr=True, enable_fallback=False, fallback_accuracy=0.0, min_required_accuracy=0.9):
        self.use_easyocr = use_easyocr
        self.enable_fallback = enable_fallback
        self.fallback_accuracy = fallback_accuracy
        self.min_required_accuracy = min_required_accuracy
        self.last_hashes = {}      # e.g. {"pot_zone": hash_val, ...}
        self.cached_values = {}    # e.g. {"pot_zone": last_value, ...}
        if use_easyocr:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False)
        else:
            import pytesseract
            self.reader = pytesseract
        # Placeholder for fallback OCR (not activated unless validated)
        self.fallback_ocr = None

    def _zone_hash(self, img):
        # Simple perceptual hash for image region
        return hashlib.md5(cv2.imencode('.png', img)[1]).hexdigest()

    def has_significant_change(self, img, zone_key):
        """Returns True if the image region for zone_key has changed."""
        h = self._zone_hash(img)
        last = self.last_hashes.get(zone_key)
        if last != h:
            self.last_hashes[zone_key] = h
            return True
        return False

    def preprocess_for_ocr(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        thresh = cv2.adaptiveThreshold(
            resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 8
        )
        return thresh

    def extract_stacks_and_pot(self, screenshot, layout, debug_info=None):
        """
        Example for extracting pot. Repeat for each zone you want to cache.
        """
        state = {}
        debug_info = debug_info if debug_info is not None else {}
        # Loop over relevant zones (example: pot, hero_stack, etc.)
        zones = {
            "pot": "pot_area",
            "hero_stack": "hero_stack"
            # ... add other zones here
        }
        for key, zone in zones.items():
            x1, y1, x2, y2 = layout.get_zone_crop(zone)
            region = screenshot[y1:y2, x1:x2]
            zone_key = f"{key}_zone"
            if not self.has_significant_change(region, zone_key):
                # No change detected, return cached
                state[key] = self.cached_values.get(zone_key)
                debug_info[zone_key] = {"cached_value": True}
                continue
            try:
                processed = self.preprocess_for_ocr(region)
                if self.use_easyocr:
                    result = self.reader.readtext(processed, detail=0)
                    text = result[0] if result else ""
                else:
                    text = self.reader.image_to_string(processed, config="--psm 7")
                    text = text.strip()
                value = self._clean_numeric(text)
                if value is not None:
                    self.cached_values[zone_key] = value
                    state[key] = value
                    debug_info[zone_key] = {"cached_value": False, "fallback_triggered": False}
                    continue
                # If primary failed, try fallback only if enabled & validated
                if self.enable_fallback and self.fallback_accuracy >= self.min_required_accuracy and self.fallback_ocr:
                    fallback_val = self.fallback_ocr(region)
                    if fallback_val is not None:
                        self.cached_values[zone_key] = fallback_val
                        state[key] = fallback_val
                        debug_info[zone_key] = {"cached_value": False, "fallback_triggered": True, "reason": "primary_invalid"}
                        continue
                # Otherwise, use last cached value
                state[key] = self.cached_values.get(zone_key)
                debug_info[zone_key] = {"cached_value": True, "fallback_triggered": False}
            except Exception as e:
                # On error, fallback to cached value
                state[key] = self.cached_values.get(zone_key)
                debug_info[zone_key] = {"cached_value": True, "error": str(e)}
        return state, debug_info

    def _clean_numeric(self, text):
        try:
            text = text.replace("$", "").replace("BB", "").replace(",", "").strip()
            # Replace common OCR errors
            text = text.replace("O", "0").replace("l", "1").replace("I", "1")
            return float(text)
        except:
            return None

