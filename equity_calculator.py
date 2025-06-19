import random
from typing import List, Tuple, Dict
import eval7


# ──────────────────────────────────────────────────────────────
#   Helpers
# ──────────────────────────────────────────────────────────────
def _str_to_card(card_str: str) -> eval7.Card:
    """'As' → eval7.Card('As')   (with ValueError guard)"""
    try:
        return eval7.Card(card_str)
    except Exception as err:
        raise ValueError(f"Bad card string: {card_str}") from err


def _parse_combo(combo: str) -> Tuple[eval7.Card, eval7.Card]:
    """
    Accepts 'QsJh' or 'Qs Jh' or 'QsJh '.
    Returns a tuple(Card, Card) in rank-then-suit order.
    """
    combo = combo.replace(" ", "").strip()
    if len(combo) != 4:
        raise ValueError(f"Combo must be 4 chars like 'QsJh'; got '{combo}'")
    return _str_to_card(combo[:2]), _str_to_card(combo[2:])


# ──────────────────────────────────────────────────────────────
#   Monte-Carlo equity engine
# ─────────────────────────────────────────────────────────────-
def calculate_equity(
    hero_cards: List[str],
    board: List[str],
    villain_range: List[str],
    *,
    num_trials: int = 5_000,
) -> float:
    """
    Monte-Carlo hero equity vs. a *range* of exact villain combos.

    hero_cards     : ['As','Kd']
    board          : ['2h','Tc','7s']  (≤ 5)
    villain_range  : ['QsJh', '9d9c', ...]  (no weighted freq yet)
    """
    hero          = [_str_to_card(c) for c in hero_cards]
    initial_board = [_str_to_card(c) for c in board]
    wins          = 0.0

    # Pre-parse villain combos once for speed
    parsed_villain: List[Tuple[eval7.Card, eval7.Card]] = [
        _parse_combo(cmb) for cmb in villain_range
    ]

    for _ in range(num_trials):
        deck = eval7.Deck()
        # remove hero + known board
        for c in hero + initial_board:
            deck.cards.remove(c)

        # --- sample villain hand that doesn't collide ---------------
        while True:
            v1, v2 = random.choice(parsed_villain)
            if v1 in deck.cards and v2 in deck.cards:
                deck.cards.remove(v1)
                deck.cards.remove(v2)
                villain = [v1, v2]
                break

        # --- complete the board ------------------------------------
        needed = 5 - len(initial_board)
        sim_board = initial_board + deck.deal(needed)

        hero_val    = eval7.evaluate(hero + sim_board)
        villain_val = eval7.evaluate(villain + sim_board)

        if hero_val > villain_val:
            wins += 1
        elif hero_val == villain_val:
            wins += 0.5
        # else villain wins -> +0

    return wins / num_trials


# ─────────────────────────────────────────────────────────────-
#   Pot-odds decision helpers
# ─────────────────────────────────────────────────────────────-
def recommend_action(
    equity: float,
    pot_odds: float,
    *,
    aggressive_threshold: float = 0.15,
) -> str:
    """
    Basic threshold logic:
    • equity < pot_odds  → FOLD
    • pot_odds ≤ equity < pot_odds+τ → CALL
    • equity ≥ pot_odds+τ → RAISE
    """
    if equity < pot_odds:
        return "FOLD"
    if equity - pot_odds < aggressive_threshold:
        return "CALL"
    return "RAISE"


def evaluate_decision(
    hero_cards: List[str],
    board: List[str],
    villain_range: List[str],
    pot_odds: float,
    *,
    num_trials: int = 5_000,
    aggressive_threshold: float = 0.15,
) -> Dict[str, float | str]:
    """
    Full pipeline: equity simulation → action recommendation.
    """
    equity = calculate_equity(hero_cards, board, villain_range, num_trials=num_trials)
    rec    = recommend_action(equity, pot_odds, aggressive_threshold=aggressive_threshold)
    return {
        "equity":        round(equity, 4),
        "pot_odds":      round(pot_odds, 4),
        "recommendation": rec,
    }


# ─────────────────────────────────────────────────────────────-
#   Quick sanity test (remove in prod)
# ─────────────────────────────────────────────────────────────-
if __name__ == "__main__":
    hero   = ["As", "Kd"]
    board  = ["2h", "Tc", "7s"]
    range_ = ["QsJh", "9d9c", "AdQd", "7c7d", "KcQc"]
    odds   = 0.34

    print(evaluate_decision(hero, board, range_, odds))
