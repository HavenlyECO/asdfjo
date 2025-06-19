# hand_logger.py  –  persistent hand / recommendation log (rev-D, 2025-06-18)
# =============================================================================

from __future__ import annotations
import csv
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class HandLogger:
    """Thread-safe hand/action logger with JSON & CSV export and autosave."""

    def __init__(
        self,
        *,
        autosave_interval: int = 60,
        autosave_file: str | Path | None = None,
        max_entries: Optional[int] = None,
    ):
        self.log: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
        self.autosave_interval = int(autosave_interval)
        self.autosave_file = Path(autosave_file).expanduser() if autosave_file else None
        self.max_entries = max_entries
        self._autosave_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        if self.autosave_file:
            self.start_autosave()

    # ──────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────
    def log_hand(
        self,
        *,
        game_state: Dict[str, Any],
        recommendations: Any,
        user_action: str | None = None,
        result: Any = None,
        hand_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "hand_id": hand_id,
            "session_id": session_id,
            "game_state": game_state,
            "recommendations": recommendations,
            "user_action": user_action,
            "result": result,
        }
        with self.lock:
            if self.max_entries and len(self.log) >= self.max_entries:
                self.log.pop(0)
            self.log.append(entry)
        print(f"[{entry['timestamp']}] Logged hand {hand_id}", flush=True)

    # ----------------------------------------------------------
    def export_json(self, filename: str | Path) -> None:
        filename = Path(filename).expanduser()
        filename.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, open(filename, "w", encoding="utf-8") as f:
            json.dump(self.log, f, indent=2, ensure_ascii=False)
        print(f"[HandLogger] JSON exported → {filename}", flush=True)

    def load_json(self, filename: str | Path) -> None:
        """Append entries from an existing JSON log (if file exists)."""
        filename = Path(filename).expanduser()
        if not filename.exists():
            return
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        with self.lock:
            self.log.extend(data)
            # enforce cap after load
            if self.max_entries and len(self.log) > self.max_entries:
                self.log = self.log[-self.max_entries :]
        print(f"[HandLogger] Loaded {len(data)} entries from {filename}", flush=True)

    # ----------------------------------------------------------
    def export_csv(self, filename: str | Path) -> None:
        def _flatten(d: Dict[str, Any], prefix="") -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k, v in d.items():
                key = f"{prefix}{k}"
                if isinstance(v, dict):
                    out.update(_flatten(v, key + "."))
                elif isinstance(v, list):
                    out[key] = json.dumps(v, ensure_ascii=False)
                else:
                    out[key] = v
            return out

        filename = Path(filename).expanduser()
        filename.parent.mkdir(parents=True, exist_ok=True)

        with self.lock:
            rows: List[Dict[str, Any]] = []
            for e in self.log:
                row = {
                    "timestamp": e["timestamp"],
                    "hand_id": e["hand_id"],
                    "session_id": e["session_id"],
                    "user_action": e["user_action"],
                    "result": json.dumps(e["result"], ensure_ascii=False)
                    if isinstance(e["result"], (dict, list))
                    else e["result"],
                }
                if isinstance(e["game_state"], dict):
                    row.update(_flatten(e["game_state"], "game_state."))
                if isinstance(e["recommendations"], dict):
                    row.update(_flatten(e["recommendations"], "recommendations."))
                else:
                    row["recommendations"] = e["recommendations"]
                rows.append(row)

        fieldnames = sorted({k for r in rows for k in r})
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[HandLogger] CSV exported → {filename}", flush=True)

    # ──────────────────────────────────────────────────────────
    #  Autosave
    # ──────────────────────────────────────────────────────────
    def start_autosave(self) -> None:
        if self._autosave_thread and self._autosave_thread.is_alive():
            return

        def _loop() -> None:
            while not self._stop_evt.wait(self.autosave_interval):
                if self.autosave_file:
                    try:
                        self.export_json(self.autosave_file)
                        print("[HandLogger] Autosave ✓", flush=True)
                    except Exception as exc:  # pragma: no cover
                        print(f"[HandLogger] Autosave failed: {exc}", flush=True)

        self._autosave_thread = threading.Thread(
            target=_loop, daemon=True, name="HandAutosave"
        )
        self._autosave_thread.start()
        print("[HandLogger] Autosave thread started.", flush=True)

    def stop_autosave(self) -> None:
        self._stop_evt.set()
        if self._autosave_thread:
            self._autosave_thread.join()

    # ──────────────────────────────────────────────────────────
    #  Context manager
    # ──────────────────────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_autosave()


# ------------------------------------------------------------------
#  Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    with HandLogger(autosave_interval=60, autosave_file="logs/autosave_hands.json") as logger:
        for i in range(3):
            logger.log_hand(
                game_state={
                    "hero_cards": ["As", "Kd"],
                    "board_cards": ["Qs", "2h", "7c"],
                    "pot": 33,
                    "villain_stats": {"vpip": 0.32, "af": 2.9},
                },
                recommendations={"GTO": "CALL", "Exploit": "RAISE"},
                user_action="CALL",
                result={"win": True, "amount": 42},
                hand_id=f"HAND_{i+1}",
                session_id="SESSION_2025-06-19",
            )
        logger.export_json("logs/session_2025-06-19.json")
        logger.export_csv("logs/session_2025-06-19.csv")
