# quality_monitor.py  –  run-time sanity checks on parsed game_state
# =================================================================

from __future__ import annotations
from typing import Any, Dict, List


def _to_list(obj: Any) -> List[str]:
    """Ensure we always work with a list of strings."""
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    # tolerate comma-separated string
    if isinstance(obj, str):
        return [x.strip() for x in obj.split(",") if x.strip()]
    return list(obj)  # last-ditch – may raise, which is fine


def check_game_state_quality(  # noqa: C901  (complexity is acceptable for this validator)
    game_state: Dict[str, Any],
    *,
    ocr_confidence: Dict[str, float] | None = None,
    card_confidence: Dict[str, float] | None = None,
) -> List[str]:
    """
    Return a list of alert strings (each suffixed with [WARNING] or [CRITICAL])
    if anything in `game_state` looks missing, implausible, or low-confidence.

    The function never raises – all problems are reported via the return value.
    """
    alerts: List[str] = []

    # ──────────────────────────────────────────────────────
    # 1. Mandatory keys present
    # ──────────────────────────────────────────────────────
    for key in {"hero_cards", "board_cards", "pot", "hero_stack", "big_blind"}:
        if game_state.get(key) in (None, "", "UNKNOWN"):
            alerts.append(f"Missing or unknown value for key: {key} [CRITICAL]")

    # ──────────────────────────────────────────────────────
    # 2. Numeric sanity
    # ──────────────────────────────────────────────────────
    for key in ("pot", "hero_stack", "big_blind"):
        val = game_state.get(key)
        if val in (None, "", "UNKNOWN"):
            continue
        try:
            num = float(val)
            if num < 0:
                alerts.append(f"Negative value for {key}: {val} [CRITICAL]")
            elif num > 1_000_000:
                alerts.append(f"Implausibly high value for {key}: {val} [WARNING]")
        except (TypeError, ValueError):
            alerts.append(f"Non-numeric value for {key}: {val} [WARNING]")

    # ──────────────────────────────────────────────────────
    # 3. OCR / card-detector confidence
    # ──────────────────────────────────────────────────────
    for field, conf in (ocr_confidence or {}).items():
        if conf < 0.70:
            alerts.append(f"OCR confidence low for {field}: {conf:.2f} [WARNING]")

    for slot, conf in (card_confidence or {}).items():
        if conf < 0.70:
            alerts.append(f"Card detection confidence low for {slot}: {conf:.2f} [WARNING]")

    # ──────────────────────────────────────────────────────
    # 4. Card-list integrity
    # ──────────────────────────────────────────────────────
    hero_cards: List[str]  = _to_list(game_state.get("hero_cards"))
    board_cards: List[str] = _to_list(game_state.get("board_cards"))
    clean_cards = [c for c in hero_cards + board_cards if c not in ("", "UNKNOWN", None)]

    if len(clean_cards) != len(set(clean_cards)):
        alerts.append("Duplicate card detected in hero or board cards [CRITICAL]")

    if set(hero_cards) & set(board_cards):
        alerts.append("Hero and board cards overlap! [CRITICAL]")

    if any(c in ("", "UNKNOWN", None) for c in hero_cards + board_cards):
        alerts.append("One or more cards missing or unknown [WARNING]")

    if len(hero_cards) not in (0, 2, 4):
        alerts.append(f"Unexpected number of hero cards: {len(hero_cards)} [WARNING]")

    if len(board_cards) > 5:
        alerts.append(f"Too many board cards detected: {len(board_cards)} [WARNING]")

    # ──────────────────────────────────────────────────────
    # 5. Fallback statistics
    # ──────────────────────────────────────────────────────
    for method, count in (game_state.get("fallback_stats") or {}).items():
        try:
            if int(count) > 2:
                alerts.append(f"Fallback method '{method}' triggered {count}× [WARNING]")
        except (TypeError, ValueError):
            alerts.append(f"Invalid fallback count for '{method}': {count} [WARNING]")

    return alerts
