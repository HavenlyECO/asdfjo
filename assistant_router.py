# assistant_router.py
# =============================================================
# Routes a GameState dictionary to one of ten specialised assistants.
#
# • NO automatic “fallback” assistant: if no rule matches, returns None.
# • Validates that every ASSISTANT_1 … ASSISTANT_10 env-var is set.
# • Uses `hero_position` if available; otherwise falls back to `position`.
# • Guards all numeric casts; avoids AttributeErrors on missing keys.
# • Routing rules documented inline for easy tweaking.
# =============================================================

from __future__ import annotations

import os
from typing import Dict, Any

# ───────────────────────────────────────────────────────
#  Load and validate assistant IDs from environment
# ───────────────────────────────────────────────────────
ASSISTANT_MAP: Dict[int, str] = {}
for i in range(1, 11):
    env_key = f"ASSISTANT_{i}"
    val = os.getenv(env_key)
    if val is None:
        raise RuntimeError(f"Environment variable {env_key} is missing")
    ASSISTANT_MAP[i] = val


# ───────────────────────────────────────────────────────

def route_to_assistant(game_state: Dict[str, Any]) -> str | None:
    """
    Decide which assistant should handle the current decision frame.

    Parameters
    ----------
    game_state : dict
        Dictionary produced by GameStateBuilder (must include at least
        'street', 'spr', 'action', and 'players').

    Returns
    -------
    str | None
        Assistant ID string if a rule fires, otherwise None.
    """

    # ---------- Extract & normalise inputs ---------------------
    street = str(game_state.get("street", "")).lower()
    spr = float(game_state.get("spr") or 0)
    action = str(game_state.get("action", "")).strip().lower()

    villain_stats = game_state.get("villain_stats", {})
    tournament = game_state.get("tournament", {})

    players_left = int(tournament.get("players_left", 6) or 6)
    bubble = bool(game_state.get("bubble", False))
    stage = str(tournament.get("stage", "")).lower()

    hero_pos = str(
        game_state.get("hero_position", game_state.get("position", ""))
    ).upper()

    is_heads_up = players_left <= 2

    # ---------- Routing rules ---------------------------------
    # 10 ▸ Bubble / ICM spots
    if bubble or stage in {"bubble", "final"}:
        return ASSISTANT_MAP[10]

    # 9 ▸ Heads-Up play
    if is_heads_up:
        return ASSISTANT_MAP[9]

    # 3 ▸ Facing big aggression (3-bet, 4-bet, jam)
    if any(word in action for word in ("3-bet", "4-bet", "jam")):
        return ASSISTANT_MAP[3]

    # 4 ▸ River bluff-catch vs. high-AF villain
    if street == "river" and float(villain_stats.get("af", 0) or 0) >= 2.5:
        return ASSISTANT_MAP[4]

    # 5 ▸ Exploitable fold-to-3bet tendency
    if float(villain_stats.get("fold_to_3b", 0) or 0) >= 65:
        return ASSISTANT_MAP[5]

    # 7 ▸ Polarised lines (check-raise, OOP check)
    if "check-raise" in action or (
        hero_pos in {"BB", "SB"} and "check" in action
    ):
        return ASSISTANT_MAP[7]

    # 6 ▸ Deep-stack solver spots (SPR ≥ 2.5 and passive line)
    if spr >= 2.5 and "check" in action:
        return ASSISTANT_MAP[6]

    # 2 ▸ Stack-size rule (SPR ≤ 2.5 or hero > 100 BB)
    if spr <= 2.5 or float(game_state.get("hero_stack", 0) or 0) > 100:
        return ASSISTANT_MAP[2]

    # 8 ▸ Generic post-flop
    if street in {"flop", "turn", "river"}:
        return ASSISTANT_MAP[8]

    # 1 ▸ Pre-flop default
    if street == "preflop":
        return ASSISTANT_MAP[1]

    # ---------- No rule matched --------------------------------
    return None
