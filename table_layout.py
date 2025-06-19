"""
Site-tuned ROI generator for ACR 6-max (and other) tables.

Main differences vs. generic layout:
• Table centre nudged lower (y ≈ 0.49).
• Radius trimmed so avatars land correctly (≈ 0.30 × min(w, h)).
• Hero card / stack offsets tightened to match live client.
• Dynamic opponent-stack offset: above the centre → stack text sits *above* avatar; below → *below* avatar.
• Everything else remains percent-based so it scales with any window-size / resolution.

Use `draw_debug_layout()` on a screenshot to eyeball the boxes and fine-tune presets if you use another site/skin.
"""
from typing import Dict, Tuple, List, Optional
import math

try:
    import cv2
    import numpy as np
except ImportError:  # head-less / server environment – only debug overlay needs these
    cv2 = None
    np  = None


class PokerTableLayout:
    """Compute percent-based Regions-of-Interest for a poker table.

    Supports 6-max, 9-max, HU, and MTT layouts.  Site-specific overrides are
    injected via the *presets* dict.
    """

    DEFAULTS = {
        # --- geometry -------------------------------------------------------
        "center": (0.50, 0.49),           # table centre (x%, y%)
        "radius": {
            "6max": 0.30,
            "9max": 0.275,
            "hu":   0.22,
            "mtt":  0.29,
        },
        # --- ROI sizes (w%, h%) -------------------------------------------
        "card_box":     (0.10, 0.11),     # hero hole-card crop
        "stack_box":    (0.09, 0.06),     # chip-count text
        "community_box":(0.30, 0.09),     # flop / turn / river bar
        "pot_box":      (0.18, 0.08),
        "action_box":   (0.21, 0.10),
        "button_box":   (0.045, 0.045),
        # --- vertical offsets ---------------------------------------------
        "hero_card_off":  -0.06,          # hole cards sit 6 % of screen height *above* hero avatar centre
        "hero_stack_off":  0.05,          # stack text ~5 % of screen height *below* avatar centre
        "action_y":       0.93,           # y-centre for the CHECK / BET buttons row
    }

    def __init__(self,
                 screen_width: int,
                 screen_height: int,
                 table_type: str = "6max",
                 presets: Optional[dict] = None):

        self.screen_width  = screen_width
        self.screen_height = screen_height
        self.table_type    = table_type.lower()

        self.seat_count = {
            "6max": 6,
            "9max": 9,
            "hu":   2,
            "mtt":  9,
        }[self.table_type]

        # merge defaults with optional site-preset
        self.cfg = self.DEFAULTS.copy()
        self.cfg["radius"] = self.DEFAULTS["radius"][self.table_type]
        if presets:
            self.cfg.update(presets)

        self.table_center = self.cfg["center"]      # (x%, y%)
        self.table_radius = self.cfg["radius"]      # scalar

        self._define_zones()

    # ------------------------------------------------------------------
    #   Seat and ROI helpers
    # ------------------------------------------------------------------
    def _seat_position(self, seat_idx: int) -> Tuple[float, float]:
        """Return seat-centre as (x%, y%).

        Seat 0 (the Hero) is fixed at the bottom of the table (90 degrees).  The
        rest are evenly spaced clockwise.
        """
        angle_step = 360 / self.seat_count
        angle_deg  = (90 + seat_idx * angle_step) % 360
        r_px       = self.table_radius * min(self.screen_width, self.screen_height)

        cx_px = self.table_center[0] * self.screen_width
        cy_px = self.table_center[1] * self.screen_height

        x_px  = cx_px + r_px * math.cos(math.radians(angle_deg))
        y_px  = cy_px + r_px * math.sin(math.radians(angle_deg))
        return x_px / self.screen_width, y_px / self.screen_height

    @staticmethod
    def _rect_from_center(cx_pct: float, cy_pct: float, w_pct: float, h_pct: float,
                          screen_w: int, screen_h: int) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) in *pixels* for a box centred at (cx_pct, cy_pct)."""
        box_w = int(screen_w * w_pct)
        box_h = int(screen_h * h_pct)
        cx_px = int(screen_w * cx_pct)
        cy_px = int(screen_h * cy_pct)

        x1 = max(0,  cx_px - box_w // 2)
        y1 = max(0,  cy_px - box_h // 2)
        x2 = min(screen_w, cx_px + box_w // 2)
        y2 = min(screen_h, cy_px + box_h // 2)
        return x1, y1, x2, y2

    # ------------------------------------------------------------------
    #   ROI definition
    # ------------------------------------------------------------------
    def _define_zones(self):
        w, h = self.screen_width, self.screen_height
        c     = self.cfg

        self.zones: Dict[str, Tuple[int, int, int, int]] = {}

        # ---- Hero -------------------------------------------------------
        hero_cx, hero_cy = self._seat_position(0)
        self.zones["hero_hole_cards"] = self._rect_from_center(
            hero_cx, hero_cy + c["hero_card_off"], *c["card_box"], w, h)
        self.zones["hero_stack"] = self._rect_from_center(
            hero_cx, hero_cy + c["hero_stack_off"], *c["stack_box"], w, h)

        # ---- Community and pot -----------------------------------------
        tbl_cx, tbl_cy = c["center"]
        self.zones["community_cards"] = self._rect_from_center(
            tbl_cx, tbl_cy - 0.06, *c["community_box"], w, h)
        self.zones["pot_area"] = self._rect_from_center(
            tbl_cx, tbl_cy - 0.14, *c["pot_box"], w, h)

        # ---- Action bar -------------------------------------------------
        self.zones["action_box"] = self._rect_from_center(
            0.5, c["action_y"], *c["action_box"], w, h)

        # ---- Dealer button ---------------------------------------------
        angle_step = 360 / self.seat_count
        btn_angle  = math.radians(90 + angle_step / 2)          # halfway between hero and seat 1
        btn_r_px   = (self.table_radius - 0.06) * min(w, h)
        cx_btn_px  = self.table_center[0] * w + btn_r_px * math.cos(btn_angle)
        cy_btn_px  = self.table_center[1] * h + btn_r_px * math.sin(btn_angle)
        self.zones["button_position"] = self._rect_from_center(
            cx_btn_px / w, cy_btn_px / h, *c["button_box"], w, h)

        # ---- Opponent stacks -------------------------------------------
        def _stack_offset(y_pct: float) -> float:
            """Top-row stacks appear *above* the avatar, bottom-row *below*."""
            return -0.06 if y_pct < self.table_center[1] else 0.07

        for seat in range(1, self.seat_count):
            cx, cy = self._seat_position(seat)
            self.zones[f"opponent_{seat}_stack"] = self._rect_from_center(
                cx, cy + _stack_offset(cy), *c["stack_box"], w, h)

    # ------------------------------------------------------------------
    #   Public helpers
    # ------------------------------------------------------------------
    def get_zone_crop(self, zone: str) -> Tuple[int, int, int, int]:
        return self.zones[zone]

    def get_all_zones(self) -> Dict[str, Tuple[int, int, int, int]]:
        return dict(self.zones)

    def seat_positions(self) -> List[Tuple[float, float]]:
        return [self._seat_position(i) for i in range(self.seat_count)]

    # ------------------------------------------------------------------
    #   Debug overlay
    # ------------------------------------------------------------------
    def draw_debug_layout(self, img):
        if cv2 is None or np is None:
            raise ImportError("OpenCV + NumPy are required for debug drawing.")
        for name, (x1, y1, x2, y2) in self.zones.items():
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        return img


# ----------------------------------------------------------------------
#   Quick test run (remove or wrap in pytest for production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    layout = PokerTableLayout(1920, 1080, "6max")
    for name, rect in layout.get_all_zones().items():
        print(f"{name:20}: {rect}")
