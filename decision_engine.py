"""
TexasSolver wrapper  –  full, self-contained patch (rev-C 2025-06-18)
===================================================================

Invokes the TexasSolver CLI and returns a single human-readable line such as
“RECOMMEND: RAISE to 12.5” or “RECOMMEND: CALL”.  Safe defaults, shell-quoting,
timeout, and robust float casting are included.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Union


# ──────────────────────────────────────────────────────────────
#  Public entry point
# ──────────────────────────────────────────────────────────────
def gto_decision_via_texassolver(
    game_state: Dict[str, Any],
    *,
    solver_path: Union[str, Path] = "./TexasSolver",
    timeout_sec: int = 4,
) -> str:
    """
    Parameters
    ----------
    game_state   Full dictionary from GameStateBuilder.
    solver_path  Path to TexasSolver executable.
    timeout_sec  Max seconds before the subprocess is killed.

    Returns
    -------
    str          “RECOMMEND: CALL” / “RECOMMEND: FOLD” / “RECOMMEND: RAISE to X”
                 or an “UNKNOWN – …” error string if solver fails.
    """

    # ----- helpers ----------------------------------------------------
    def _csv(cards: List[str]) -> str:
        return ",".join(cards) if cards else ""

    def _f(x: Any, default: float = 0.0) -> float:
        "Cast to float safely."
        try:
            return float(x)
        except Exception:
            return default

    # ----- extract from game_state -----------------------------------
    board_cards:  List[str] = game_state.get("board_cards", [])
    hero_cards:   List[str] = game_state.get("hero_cards", [])
    villain_rng:  List[str] = game_state.get("villain_range", [])

    pot         = _f(game_state.get("pot"))
    hero_stack  = _f(game_state.get("hero_stack"))
    hero_seat   = game_state.get("hero_position", "SB")

    bb_stack = _f(game_state.get("players", {}).get("BB", {}).get("stack"))
    if bb_stack == 0.0:  # fallback to largest villain stack
        bb_stack = max(
            (_f(p.get("stack"))
             for seat, p in game_state.get("players", {}).items()
             if seat != hero_seat),
            default=0.0,
        )

    # ----- build CLI --------------------------------------------------
    mode = "postflop" if board_cards else "preflop"
    exe  = str(Path(solver_path).expanduser())

    cmd: List[str] = [
        exe,
        "--mode",          mode,
        "--hero-cards",    _csv(hero_cards),
        "--board",         _csv(board_cards),
        "--villain-range", _csv(villain_rng) or "random",
        "--pot",           f"{pot}",
        "--stacks",        f"{hero_stack},{bb_stack}",
        "--json",          # tell solver to emit JSON
    ]
    cmd_q = [shlex.quote(arg) for arg in cmd]

    # ----- run solver -------------------------------------------------
    try:
        proc = subprocess.run(
            cmd_q,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=True,
        )
        solver_out = json.loads(proc.stdout)
    except FileNotFoundError:
        return "RECOMMEND: UNKNOWN – TexasSolver binary not found"
    except subprocess.TimeoutExpired:
        return "RECOMMEND: UNKNOWN – TexasSolver timed-out"
    except (subprocess.CalledProcessError, json.JSONDecodeError) as err:
        return f"RECOMMEND: UNKNOWN – solver failure ({err})"

    # ----- choose EV-max action --------------------------------------
    ev = {
        "FOLD":  _f(solver_out.get("fold_ev"),  float("-inf")),
        "CALL":  _f(solver_out.get("call_ev"),  float("-inf")),
        "RAISE": _f(solver_out.get("raise_ev"), float("-inf")),
        "BET":   _f(solver_out.get("bet_ev"),   float("-inf")),
    }
    best = max(ev, key=ev.get)

    if best in {"RAISE", "BET"}:
        size_key = "raise_size" if best == "RAISE" else "bet_size"
        size_val = solver_out.get(size_key)
        return f"RECOMMEND: {best} to {size_val}" if size_val else f"RECOMMEND: {best}"

    return f"RECOMMEND: {best}"
