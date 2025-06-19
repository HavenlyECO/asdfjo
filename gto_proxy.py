# gto_proxy.py  –  full patched version (rev-B, 2025-06-18)
# =========================================================
# Offline training:
#     python gto_proxy.py --train solver_data.json --out gto_proxy_model.pkl
#
# Live usage:
#     from gto_proxy import fast_gto_proxy
#     rec = fast_gto_proxy(game_state)   # → "RECOMMEND: CALL" / …
#
# This file includes:
#   • Data-loader + DecisionTree training (with LabelEncoder)           (_load_training_data, train_and_save_model)
#   • Safe runtime loader for the pickled proxy                        (_load_proxy)
#   • Feature extractor that enforces training order                   (_extract_features)
#   • Pre-flop bucketed lookup table + ML fallback                     (fast_gto_proxy)
#   • CLI guard so it doubles as a trainer script

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


# ──────────────────────────────────────────────────────────────
#  Offline training helpers
# ──────────────────────────────────────────────────────────────
def _load_training_data(path: Path) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Read solver JSON → feature matrix X, encoded label vector y, label-encoder."""
    data = json.loads(path.read_text())
    X, y = [], []
    for row in data:
        X.append(
            [
                int(row.get("street", 0)),
                float(row.get("hero_stack", 0)),
                float(row.get("bb_stack", 0)),
                float(row.get("pot", 0)),
                int(row.get("position", 0)),
                int(row.get("action", 0)),
            ]
        )
        y.append(row["best_action"])
    X_arr = np.asarray(X, dtype=float)
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    return X_arr, y_enc, le


def train_and_save_model(
    data_path: str = "solver_data.json",
    model_path: str = "gto_proxy_model.pkl",
) -> None:
    """Train DecisionTree on solver data and pickle {'model': clf, 'encoder': le}."""
    X, y, le = _load_training_data(Path(data_path))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(max_depth=8, random_state=42)
    clf.fit(X_tr, y_tr)

    print("Validation accuracy:", round(clf.score(X_te, y_te), 3))
    joblib.dump({"model": clf, "encoder": le}, model_path)
    print("Model saved to", model_path)


# ──────────────────────────────────────────────────────────────
#  Runtime proxy — load tree + encoder once
# ──────────────────────────────────────────────────────────────
def _load_proxy(model_path: str = "gto_proxy_model.pkl"):
    try:
        bundle = joblib.load(model_path)
        return bundle["model"], bundle["encoder"]
    except Exception as err:
        warnings.warn(f"GTO proxy model not loaded ({err}) – falling back.")
        return None, None


_MODEL, _ENCODER = _load_proxy()


def _extract_features(gs: dict) -> np.ndarray:
    """Convert live game_state → feature vector; raise if any key missing."""
    try:
        feats = np.array(
            [
                int(gs["street"]),
                float(gs["hero_stack"]),
                float(gs["bb_stack"]),
                float(gs["pot"]),
                int(gs["position"]),
                int(gs["action"]),
            ],
            dtype=float,
        )
    except KeyError as k:
        raise ValueError(f"Missing game_state field {k}") from None
    return feats.reshape(1, -1)


# ──────────────────────────────────────────────────────────────
#  Pre-flop lookup table (bucketed)
# ──────────────────────────────────────────────────────────────
# key = (position_idx, bb_bucket, action_code)
# bb_bucket = min(stack_bb // 20, 5)  → 0: <20bb, 1: 20-39 … 5: 100+ bb
_PREFLOP_CHART = {
    (2, 5, 0): "RAISE",  # UTG, 100bb+, unopened pot
    # … extend with your chart …
}


# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────
def fast_gto_proxy(game_state: dict) -> str:
    """
    Return instant GTO-style recommendation using:
      1. Hard-coded pre-flop chart (if street == 0).
      2. Decision-tree proxy trained from solver output.
      3. 'UNKNOWN' if neither applies.
    """

    # ----- Chart lookup ---------------------------------------
    if game_state.get("street") == 0:  # pre-flop
        key = (
            int(game_state.get("position", 0)),
            min(int(game_state.get("hero_stack", 0)) // 20, 5),
            int(game_state.get("action", 0)),
        )
        if key in _PREFLOP_CHART:
            return f"RECOMMEND: {_PREFLOP_CHART[key]}"

    # ----- ML proxy -------------------------------------------
    if _MODEL is not None and _ENCODER is not None:
        try:
            feats = _extract_features(game_state)
            pred_idx = _MODEL.predict(feats)[0]
            rec = _ENCODER.inverse_transform([pred_idx])[0]
            return f"RECOMMEND: {rec}"
        except Exception as err:
            print(f"[GTO proxy] {err}", file=sys.stderr)

    return "RECOMMEND: UNKNOWN"


# ──────────────────────────────────────────────────────────────
#  CLI entry-point for offline training
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GTO proxy model")
    parser.add_argument("--train", metavar="solver_data.json")
    parser.add_argument("--out", default="gto_proxy_model.pkl")
    args = parser.parse_args()

    if args.train:
        train_and_save_model(args.train, args.out)
    else:
        print("Use --train solver_data.json to build the model.")
