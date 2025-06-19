"""
decision_engine.py – rev-F 2025-06-18
=====================================
*  **ICM layer integrated.**  The engine now evaluates four independent engines
   every frame and automatically considers ICM equity when the hand is in a
   tournament (payouts list present):

      1. TexasSolver          → EV gap   (confidence)
      2. LLM assistant        → log-prob (confidence)
      3. EHS Monte-Carlo      → |EHS-PO| (confidence)
      4. ICM equity scenarios → ΔICM     (confidence)

*  Arbitration: solver auto-wins when its EV gap ≥ 0.02 × pot; otherwise each
   engine votes with its confidence and the highest sum wins.
*  If no tournament info exists the ICM engine returns (None, 0) and is simply
   ignored.
"""

from __future__ import annotations
import json, random, sys, math
from typing import Dict, Any, Optional, Tuple
import eval7

# ───────────────────────────────────────────────────────
#  Config knobs
# ────────────────────────────────────────────────────
_SOLVER_MIN_GAP = 0.02   # pot‑EV edge needed to auto‑choose solver output
_EHS_ITERS      = 2_000
_LLM_FALLBACK_CP= 0.15   # assumed confidence if API gives no logprobs

# ───────────────────────────────────────────────────────
#  1) TexasSolver wrapper (returns rec, confidence)
# ────────────────────────────────────────────────────

def _solver_layer(gs: Dict[str, Any]) -> Tuple[Optional[str], float]:
    try:
        from solver_integration import gto_decision_via_texassolver  # type: ignore
        rec = gto_decision_via_texassolver(gs)
        if not rec.startswith("RECOMMEND"):
            return None, 0.0
        meta = gs.get("solver_meta", {})
        gap  = float(meta.get("ev_gap", 0))
        pot  = float(gs.get("pot", 1) or 1)
        return rec, max(gap / pot, 0.0)
    except Exception as err:
        print(f"[solver] {err}", file=sys.stderr)
        return None, 0.0

# ────────────────────────────────────────────────────
#  2) LLM assistant layer (returns rec, confidence)
# ───────────────────────────────────────────────────

def _llm_layer(gs: Dict[str, Any]) -> Tuple[Optional[str], float]:
    try:
        from assistant_router import route_to_assistant  # type: ignore
        from openai_api_runner import call_assistant_with_prompt  # type: ignore
        aid = route_to_assistant(gs)
        if not aid:
            return None, 0.0
        prompt = _prompt(gs) + "\nRespond with FOLD, CALL, RAISE <size>, or BET <size>."
        raw, logp = call_assistant_with_prompt(aid, prompt, return_logprobs=True)
        rec = _parse_llm(raw)
        if rec.startswith("RECOMMEND"):
            conf = _llm_conf(logp)
            return rec, conf
    except Exception as err:
        print(f"[LLM] {err}", file=sys.stderr)
    return None, 0.0


def _llm_conf(lp: Optional[float]) -> float:
    if lp is None:  # API didn’t return log‑prob
        return _LLM_FALLBACK_CP
    return max(min(math.exp(lp), 1.0), 0.0)

# ───────────────────────────────────────────────────
#  3) EHS equity layer (returns rec, confidence)
# ───────────────────────────────────────────────────

def _ehs_layer(gs: Dict[str, Any]) -> Tuple[Optional[str], float]:
    try:
        hole  = "".join(gs["hero_cards"])
        board = "".join(gs["board_cards"])
        spr   = float(gs["spr"])
        pos   = gs["hero_position"].upper()
    except KeyError:
        return None, 0.0

    ehs = _calc_ehs(hole, board, _EHS_ITERS)
    pot_odds = float(gs.get("pot_odds", 0))
    conf = abs(ehs - pot_odds)

    if spr < 1.0 and ehs > 0.40:
        return "RECOMMEND: RAISE to ALL-IN", conf
    if ehs > 0.80:
        return "RECOMMEND: VALUE-BET 75% pot", conf
    if 0.55 <= ehs <= 0.80 and pos in {"BTN", "CO"}:
        return "RECOMMEND: CALL / POT-CONTROL", conf
    if ehs < 0.25 and _board_dry(board) and pos in {"BTN", "CO"}:
        return "RECOMMEND: BLUFF 66% pot", conf
    return None, 0.0


def _calc_ehs(hole: str, board: str, iters: int) -> float:
    hero = [eval7.Card(hole[:2]), eval7.Card(hole[2:])]
    board_cards = [eval7.Card(board[i : i + 2]) for i in range(0, len(board), 2)]
    dead = set(hero + board_cards)
    deck = [c for c in eval7.Deck() if c not in dead]

    wins = ties = 0
    needed = 7 - len(hero) - len(board_cards)
    for _ in range(iters):
        random.shuffle(deck)
        draw = deck[:needed]
        full_board = board_cards + draw
        opp = draw[:2]
        hval = eval7.evaluate(hero + full_board)
        oval = eval7.evaluate(opp + full_board)
        if hval < oval:
            wins += 1
        elif hval == oval:
            ties += 1
    return (wins + 0.5 * ties) / iters

# ─────────────────────────────────────────────────────────────-
#  ICM layer
# ─────────────────────────────────────────────────────────────-

def _icm_layer(gs: Dict[str, Any]) -> Tuple[Optional[str], float]:
    try:
        from poker_icm import calculate_icm_equity  # type: ignore
    except ImportError:
        return None, 0.0

    tmeta = gs.get("tournament", {})
    payouts = tmeta.get("payouts")
    if not payouts:  # not an MTT hand
        return None, 0.0

    players = gs.get("players", {})
    names   = list(players.keys())
    stacks  = [float(p.get("stack", 0)) for p in players.values()]
    hero_pos = gs.get("hero_position", names[0])
    hero_idx = names.index(hero_pos) if hero_pos in names else 0
    hero_stack = stacks[hero_idx]

    to_call = float(players.get(hero_pos, {}).get("amount", 0))
    risk = min(hero_stack, to_call) if to_call else hero_stack

    def _eq(new_stack):
        tmp = stacks.copy(); tmp[hero_idx] = new_stack
        return calculate_icm_equity(tmp, payouts)[hero_idx]

    eq_fold = _eq(hero_stack)
    eq_call_win = _eq(hero_stack + risk)
    eq_call_lose= _eq(max(0, hero_stack - risk))

    conf = abs(eq_call_win - eq_fold)

    if eq_call_win > eq_fold and eq_call_win > eq_call_lose:
        return "RECOMMEND: CALL", conf
    if eq_fold >= eq_call_win and eq_fold >= eq_call_lose:
        return "RECOMMEND: FOLD", conf
    return None, 0.0


def _board_dry(board: str) -> bool:
    ranks = board[::2]; suits = board[1::2];
    return len(set(ranks)) == 3 and len(set(suits)) == 3

# ───────────────────────────────────────────────────
#  Arbiter
# ───────────────────────────────────────────────────

def decision_engine(gs: Dict[str, Any]) -> str:
    engines = {
        "solver": _solver_layer(gs),
        "llm":    _llm_layer(gs),
        "ehs":    _ehs_layer(gs),
        "icm":    _icm_layer(gs),
    }

    # auto‑pick if solver gap is large
    rec_solver, conf_solver = engines["solver"]
    if conf_solver >= _SOLVER_MIN_GAP and rec_solver:
        return rec_solver

    # consensus check
    votes = {}
    for name, (rec, conf) in engines.items():
        if rec:
            votes.setdefault(rec, 0)
            votes[rec] += conf
    if votes:
        best_rec = max(votes, key=votes.get)
        return best_rec

    return "RECOMMEND: UNKNOWN"

# ───────────────────────────────────────────────────
#  Minimal helpers for LLM prompt / parse
# ───────────────────────────────────────────────────

def _prompt(gs: Dict[str, Any]) -> str:
    slim = {k: gs[k] for k in ("street", "spr", "action", "pot_odds") if k in gs}
    return f"GameState:\n{json.dumps(slim, indent=2)}"


def _parse_llm(raw: str) -> str:
    txt = raw.strip().upper()
    for k in ("FOLD", "CALL", "RAISE", "BET"):
        if txt.startswith(k):
            return f"RECOMMEND: {txt.title()}"
    return "RECOMMEND: UNKNOWN"

