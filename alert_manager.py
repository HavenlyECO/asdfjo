# alert_manager.py  –  logging + optional TTS / overlay
# =====================================================

from __future__ import annotations
import logging
from pathlib import Path
from typing import Callable

try:
    import pyttsx3

    _TTS = pyttsx3.init()
except ImportError:  # pragma: no cover
    _TTS = None


class AlertManager:
    """
    Small helper class that
      • prints / logs alerts,
      • appends them to a text file,
      • speaks CRITICAL messages (if pyttsx3 installed),
      • optionally shows them via a GUI overlay callback.
    """

    LOG_PATH = Path("logs/poker_assistant_alerts.log").expanduser()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # GUI overlay callback; set from outside if you have a HUD
    overlay_callback: Callable[[str, str], None] = staticmethod(lambda msg, lvl: None)

    # ------------------------------------------------------------------
    @classmethod
    def _write_to_file(cls, line: str) -> None:
        cls.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with cls.LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # ------------------------------------------------------------------
    @classmethod
    def alert(cls, msg: str, level: str = "WARNING") -> None:
        """
        Emit an alert. `level` should be DEBUG / INFO / WARNING / CRITICAL.
        """
        level = level.upper()
        text = f"{level}: {msg}"

        # Console
        print(text, flush=True)

        # Plain text log file
        cls._write_to_file(text)

        # Standard logging module
        logging.log(getattr(logging, level, logging.WARNING), msg)

        # Speech (critical only)
        if level == "CRITICAL" and _TTS:
            try:
                _TTS.say(msg)
                _TTS.runAndWait()
            except Exception:  # pragma: no cover
                pass

        # Optional overlay
        try:
            cls.overlay_callback(msg, level)
        except Exception:  # pragma: no cover
            pass

