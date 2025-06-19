# frame_change_detector.py  –  rev-C (2025-06-18)
# ==========================================================
# Hybrid dHash + edge-diff detector with bounded cache.

from __future__ import annotations
import cv2
import numpy as np
from collections import OrderedDict
from typing import Dict, Optional


class FrameChangeDetector:
    """
    Detect significant visual change between frames/ROIs.

    Parameters
    ----------
    threshold   : float   Edge-pixel ratio that triggers “change”. 0.03 ≈ 3 %
    blur_ksize  : int     Gaussian blur ksize before edge diff.
    hash_size   : int     dHash grid width → returns hash_size**2 bits.
    max_cache   : int     Maximum #region hashes to keep (FIFO).
    """

    def __init__(
        self,
        threshold: float = 0.02,
        *,
        blur_ksize: int = 3,
        hash_size: int = 8,
        max_cache: int = 128,
    ):
        if threshold < 0.005:
            raise ValueError("threshold too small")
        self.threshold  = threshold
        self.blur_ksize = blur_ksize
        self.hash_size  = hash_size
        self.max_cache  = max_cache

        self._hash_cache: "OrderedDict[str,int]" = OrderedDict()
        self._png_cache:  "OrderedDict[str,bytes]" = OrderedDict()

    # ──────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────
    def has_significant_change(self, region: str, img: np.ndarray) -> bool:
        """
        Return True iff new `img` for `region` differs materially from last one.
        Side-effect: caches new hash+PNG if change detected OR first visit.
        """

        h = self._dhash_int(img)
        if self._hash_cache.get(region) == h:
            return False

        old_bytes = self._png_cache.get(region)
        if old_bytes is not None:
            old_img = cv2.imdecode(np.frombuffer(old_bytes, np.uint8), cv2.IMREAD_COLOR)
            if old_img is not None:
                # resize if shapes differ
                if old_img.shape != img.shape:
                    old_img = cv2.resize(old_img, (img.shape[1], img.shape[0]))
                if not self._edge_diff_change(old_img, img):
                    return False  # difference too small

        self._update_cache(region, h, img)
        return True

    def reset(self, region: Optional[str] = None) -> None:
        if region is None:
            self._hash_cache.clear()
            self._png_cache.clear()
        else:
            self._hash_cache.pop(region, None)
            self._png_cache.pop(region, None)

    # ──────────────────────────────────────────────────
    #  Internals
    # ──────────────────────────────────────────────────
    def _update_cache(self, region: str, h: int, img: np.ndarray) -> None:
        if len(self._hash_cache) >= self.max_cache:
            self._hash_cache.popitem(last=False)
            self._png_cache.popitem(last=False)
        self._hash_cache[region] = h
        self._png_cache[region]  = cv2.imencode(".png", img)[1].tobytes()

    def _dhash_int(self, img: np.ndarray) -> int:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.hash_size + 1, self.hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        bits = "".join("1" if b else "0" for b in diff.flatten())
        return int(bits, 2)

    def _edge_diff_change(self, a: np.ndarray, b: np.ndarray) -> bool:
        g1 = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (self.blur_ksize,)*2, 0)
        g2 = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (self.blur_ksize,)*2, 0)
        e1 = cv2.Canny(g1, 30, 90)
        e2 = cv2.Canny(g2, 30, 90)
        diff = cv2.absdiff(e1, e2)
        ratio = np.count_nonzero(diff) / diff.size
        return ratio >= self.threshold


# ──────────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    det = FrameChangeDetector(threshold=0.03)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if det.has_significant_change("cam", frame):
                print("Change!")
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
