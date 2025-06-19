from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict, Any, Deque, Tuple


class VillainProfiler:
    """
    Tracks per-villain statistics (VPIP, PFR, AF, etc.).
    Call `update()` once per *action*; pass `new_hand=True`
    on the first action you observe from that villain each hand.
    """

    def __init__(self, max_hand_history: int = 100):
        self.stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                # hand-level counters
                "hands_played": 0,
                "vpip_hands": 0,
                "pfr_hands": 0,
                "showdowns": 0,
                "3bet_opportunities": 0,
                "folds_to_3b": 0,
                # action counters (occurrence-based)
                "bets": 0,
                "raises": 0,
                "calls": 0,
                # recent history
                "actions": deque(maxlen=max_hand_history)  # type: Deque[Tuple[str, str]]
            }
        )

    # ──────────────────────────────────────────────────────────
    def update(
        self,
        seat: str,
        action: str,
        *,
        street: str = "",
        new_hand: bool = False,
        voluntary: bool = False,
        is_3b_spot: bool = False,
        did_fold_to_3b: bool = False,
        saw_showdown: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        seat             : villain identifier
        action           : 'bet', 'raise', 'call', 'fold', etc.
        street           : 'preflop', 'flop', 'turn', 'river' (optional)
        new_hand         : True on villain's first action of a new hand
        voluntary        : True if action counts as VPIP for this hand
        is_3b_spot       : Villain faced a 3-bet pre-flop this hand
        did_fold_to_3b   : Villain folded to that 3-bet
        saw_showdown     : Villain reached showdown this hand
        """

        st = self.stats[seat]

        # increment per-hand counters once
        if new_hand:
            st["hands_played"] += 1
            if is_3b_spot:
                st["3bet_opportunities"] += 1
            if saw_showdown:
                st["showdowns"] += 1

        # VPIP / PFR (flagged per hand)
        if street == "preflop" and voluntary:
            if st.get("_vpip_mark") != st["hands_played"]:
                st["vpip_hands"] += 1
                st["_vpip_mark"] = st["hands_played"]
            if action in {"bet", "raise"} and st.get("_pfr_mark") != st["hands_played"]:
                st["pfr_hands"] += 1
                st["_pfr_mark"] = st["hands_played"]

        # action counters
        match action:
            case "bet":
                st["bets"] += 1
            case "raise":
                st["raises"] += 1
            case "call":
                st["calls"] += 1

        # fold-to-3bet
        if new_hand and is_3b_spot and did_fold_to_3b:
            st["folds_to_3b"] += 1

        # push to recent history
        st["actions"].append((street or "?", action))

    # ──────────────────────────────────────────────────────────
    def get_profile(self, seat: str) -> Dict[str, float | list]:
        st = self.stats[seat]
        hp = max(st["hands_played"], 1)
        vpip = st["vpip_hands"] / hp
        pfr  = st["pfr_hands"] / hp
        af   = (st["bets"] + st["raises"]) / max(st["calls"], 1)
        ft3b = st["folds_to_3b"] / max(st["3bet_opportunities"], 1)
        sd   = st["showdowns"] / hp

        return {
            "hands_played": hp,
            "vpip": round(vpip, 3),
            "pfr": round(pfr, 3),
            "af": round(af, 2),
            "fold_to_3b": round(ft3b, 3),
            "sd_freq": round(sd, 3),
            "recent_actions": list(st["actions"]),
        }

    # ──────────────────────────────────────────────────────────
    def reset(self, seat: str) -> None:
        """Clear all stats for a given villain (e.g. table change)."""
        self.stats.pop(seat, None)
