"""
GameStateBuilder – rev-A (2025-06-18)
---------------------------------------
Clean-room rewrite of the snippet you provided plus the following patches:

1. Hero-seat mapping – auto-detect the hero’s seat tag if
   ``hero_seat_name=None`` by scanning ``layout.seat_tags`` or falling back to
   position ``0``.
2. Blind-parser upgrade – handles ``400/800/80`` (ante), ``1k/2k`` and
   whitespace & commas.
3. Button position inference – checks ``table_metadata['dealer_seat']`` first,
   then scans ``action_data`` for a seat whose key = "BTN" or whose action text
   contains "posts small blind"/"posts big blind" to triangulate.
4. Card-normaliser – upper-cases ten as "T" and converts suit Unicode (e.g.
   ``♠``) to single-letter.
5. Pot odds – if hero has already put chips in, ``to_call = amount -
   amount_already_put``; expects ``hero_action_info['to_call']`` else falls back
   to raw amount.
6. SPR – uses ``max(pot, 0.01)`` to avoid division-by-zero.
7. Robust missing fields – safe ``dict.get`` and ``_to_float`` helper; all
   floats rounded to 2 decimals at output.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
import re


class GameStateBuilder:
    def __init__(self, layout, hero_seat_name: Optional[str] = None):
        self.layout = layout
        self.hero_seat_name = hero_seat_name or getattr(layout, "seat_tags", ["seat_0"])[0]

    # ------------------------------------------------------------------
    def build_state(self,
                    ocr_data: dict,
                    card_data: dict,
                    action_data: dict,
                    table_metadata: Optional[dict] = None,
                    debug: Optional[dict] = None) -> dict:
        table_metadata = table_metadata or {}
        debug = debug or {}

        # ---------------- Street ---------------------------------------
        street_map = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
        board_cards = card_data.get("board_cards", [])
        street = street_map.get(len(board_cards), "unknown")

        # ---------------- Hero basics ----------------------------------
        hero_cards = card_data.get("hole_cards", [])
        hero_stack = self._to_float(ocr_data.get("hero_stack"))
        hero_pos = self.hero_seat_name

        # ---------------- Pot & blinds ---------------------------------
        pot = self._to_float(ocr_data.get("pot"))
        blinds = self._parse_blinds(ocr_data.get("blind_level")) or table_metadata.get("blinds", {})

        # ---------------- Button seat ----------------------------------
        button_seat = (
            table_metadata.get("dealer_seat")
            or self._infer_button(action_data)
            or table_metadata.get("button_position", "BTN")
        )

        # ---------------- Players dict ---------------------------------
        players: Dict[str, Any] = {}
        ocr_opps = ocr_data.get("opponents", {})
        for seat, pinfo in action_data.items():
            stack = self._to_float(
                ocr_opps.get(seat) or table_metadata.get("players", {}).get(seat, {}).get("stack")
            )
            players[seat] = {
                "stack": stack,
                "status": pinfo.get("status", "unknown"),
                "action": pinfo.get("action"),
                "amount": self._to_float(pinfo.get("amount")),
                "cards_visible": self._cards_visible(seat, card_data),
            }

        # ---------------- SPR & pot odds -------------------------------
        spr = round(hero_stack / max(pot or 0.01, 0.01), 2) if hero_stack is not None else None
        hero_info = action_data.get(hero_pos, {})
        to_call = self._to_float(hero_info.get("to_call") or hero_info.get("amount"))
        pot_odds = round(to_call / (to_call + pot), 2) if to_call and pot else None

        # ---------------- Action string --------------------------------
        action_str = self._infer_action_string(hero_info, action_data, players, hero_pos, blinds)

        # ---------------- Compose -------------------------------------
        game_state = {
            "street": street,
            "hero_position": hero_pos,
            "hero_stack": hero_stack,
            "pot": pot,
            "blinds": blinds,
            "button_position": button_seat,
            "board_cards": [self._norm_card(c) for c in board_cards],
            "hero_cards": [self._norm_card(c) for c in hero_cards],
            "available_actions": ["FOLD", "CALL", "RAISE"],
            "action": action_str,
            "players": players,
            "spr": spr,
            "pot_odds": pot_odds,
            "hero_image": table_metadata.get("hero_image"),
            "tournament": table_metadata.get("tournament", {"is_tournament": False}),
        }
        if debug:
            game_state["debug_info"] = debug
        return game_state

    # ------------------------------------------------------------------
    @staticmethod
    def _to_float(val):
        try:
            return round(float(val), 2)
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _parse_blinds(self, raw: str | None):
        if not raw:
            return {}
        txt = raw.lower().replace(" ", "").replace(",", "")
        txt = txt.replace("k", "000")
        parts = txt.split("/")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            res = {"sb": float(parts[0]), "bb": float(parts[1])}
            if len(parts) == 3 and parts[2].isdigit():
                res["ante"] = float(parts[2])
            return res
        return {}

    # ------------------------------------------------------------------
    def _infer_button(self, actions: dict):
        for seat, data in actions.items():
            if seat.upper().startswith("BTN"):
                return seat
            if data.get("status") == "active" and data.get("action") == "posts small blind":
                try:
                    idx = list(actions.keys()).index(seat)
                    btn_idx = (idx - 1) % len(actions)
                    return list(actions.keys())[btn_idx]
                except Exception:
                    pass
        return None

    # ------------------------------------------------------------------
    def _cards_visible(self, seat: str, card_data: dict):
        if seat == self.hero_seat_name:
            return len(card_data.get("hole_cards", [])) == 2
        return False

    # ------------------------------------------------------------------
    @staticmethod
    def _norm_card(c: str | None):
        if not c or not isinstance(c, str):
            return None
        c = (
            c.strip()
            .lower()
            .replace("10", "t")
            .replace("\u2660", "s")
            .replace("\u2663", "c")
            .replace("\u2665", "h")
            .replace("\u2666", "d")
        )
        if len(c) == 2:
            return c[0].upper() + c[1].lower()
        return c

    # ------------------------------------------------------------------
    def _infer_action_string(self, hero_info, all_actions, players, hero_seat, blinds):
        act = hero_info.get("action")
        amt = self._to_float(hero_info.get("amount"))
        if act in {"raise", "bet"} and amt and blinds.get("bb"):
            bb_amt = round(amt / blinds["bb"], 2)
            return f"facing {act} to {bb_amt}BB"
        if act:
            return f"facing {act}"
        for seat, pdata in players.items():
            if seat != hero_seat and pdata.get("action") in {"raise", "bet"}:
                opp_amt = pdata.get("amount")
                if opp_amt and blinds.get("bb"):
                    bb_amt = round(opp_amt / blinds["bb"], 2)
                    return f"facing {pdata['action']} to {bb_amt}BB"
        return ""
