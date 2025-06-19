"""
decision_engine.py – rev-E 2025-06-18
====================================
• **Lightweight heuristics removed entirely.**  No rule‑of‑thumb logic remains.
• Three independent evaluators now run in parallel:
      1. **TexasSolver**  → supplies EVs + best action.
      2. **LLM assistant**→ provides natural‑language recommendation + model logprobs.
      3. **EHS module**   → Monte‑Carlo equity → rule‑based action.
• A simple arbiter chooses the answer with the **highest confidence score**:
      * Solver confidence   = EV_gap / pot (≥ 0 when best beats 2nd‑best).
      * LLM confidence      = (max logprob – second) if the API returns it, else 0.15.
      * EHS confidence      = |EHS − PotOdds|.
• If two engines agree, their scores are summed → higher priority.
• Returns "RECOMMEND: UNKNOWN" only if every engine fails.

Change these constants to tune behaviour:
    _SOLVER_MIN_GAP   = 0.02      # pot‑EV gap for solver to “win” automatically
    _EHS_ITERS        = 2_000
    _LLM_FALLBACK_CP  = 0.15      # default confidence when logprobs unavailable

Dependencies: eval7, solver_integration, assistant_router, openai_api_runner.
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
        out = gto_decision_via_texassolver(gs)
        if not out.startswith("RECOMMEND"):  # solver failed
            return None, 0.0
        # EV gap heuristic
        ev_best  = float(gs.get("solver_meta", {}).get("ev_best", 0))
        ev_second= float(gs.get("solver_meta", {}).get("ev_second", 0))
        gap = ev_best - ev_second
        pot = float(gs.get("pot", 1)) or 1.0
        conf = max(gap / pot, 0.0)
        return out, conf
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
        prompt = _prompt_from_state(gs)
        prompt += "\nRespond with exactly one action: FOLD, CALL, RAISE <size>, or BET <size>."
        raw, logprobs = call_assistant_with_prompt(aid, prompt, return_logprobs=True)
        rec = _parse_llm(raw)
        if rec.startswith("RECOMMEND"):
            conf = _llm_confidence(logprobs)
            return rec, conf
        return None, 0.0
    except Exception as err:
        print(f"[LLM] {err}", file=sys.stderr)
        return None, 0.0


def _llm_confidence(logp: Optional[float]) -> float:
    if logp is None:
        return _LLM_FALLBACK_CP
    return max(min(math.exp(logp), 1.0), 0.0)

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

def _prompt_from_state(gs: Dict[str, Any]):
    slim = {k: gs[k] for k in ("street", "spr", "action", "pot_odds") if k in gs}
    return f"GameState:\n{json.dumps(slim, indent=2)}"


def _parse_llm(raw: str) -> str:
    txt = raw.strip().upper()
    for k in ("FOLD", "CALL", "RAISE", "BET"):
        if txt.startswith(k):
            return f"RECOMMEND: {txt.title()}"
    return "RECOMMEND: UNKNOWN"

