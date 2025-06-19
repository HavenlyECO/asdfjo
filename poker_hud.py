# poker_hud.py – frameless always-on-top overlay (rev-C, 2025-06-18)
# ================================================================

from PyQt5 import QtWidgets, QtCore, QtGui
import html, sys


class PokerHUD(QtWidgets.QWidget):
    # Cross-thread signal
    state_updated = QtCore.pyqtSignal(dict, str)

    def __init__(self) -> None:
        super().__init__()

        # Window flags: frameless, click-through, on-top, no focus steal
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)

        # Label
        self.label = QtWidgets.QLabel(self)
        self.label.setStyleSheet(
            "color:white; background:transparent; font-size:18pt; font-weight:600;"
        )
        self.label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # Drop-shadow
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(6)
        shadow.setColor(QtGui.QColor(0, 0, 0, 160))
        shadow.setOffset(2, 2)
        self.label.setGraphicsEffect(shadow)

        # Wire signal
        self.state_updated.connect(self.update_state)

        # Default position
        geo = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.move(geo.x() + 10, geo.y() + 10)

        # Whole-widget transparency style
        self.setStyleSheet("background:transparent;")

    # Slot
    @QtCore.pyqtSlot(dict, str)
    def update_state(self, gs: dict, rec: str) -> None:
        hero  = html.escape(" ".join(gs.get("hero_cards", [])))
        board = html.escape(" ".join(gs.get("board_cards", [])))
        pot   = gs.get("pot", 0)
        stack = gs.get("hero_stack", 0)
        vill  = gs.get("villain_stats", {})
        vpip  = vill.get("vpip", 0.0)
        af    = vill.get("af", 0.0)

        text = (
            f"Hero: {hero}<br>"
            f"Board: {board}<br>"
            f"Pot: {pot}&nbsp;&nbsp;Stack: {stack}<br>"
            f"Villain&nbsp;VPIP: {vpip:.2f}&nbsp;&nbsp;AF: {af:.2f}<br>"
            f"<b>{html.escape(rec)}</b>"
        )
        self.label.setText(text)
        self.adjustSize()

    def move_to(self, x: int, y: int) -> None:
        self.move(x, y)


# ------------------------------------------------------------------
if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)

    hud = PokerHUD()
    hud.show()

    # demo update after 2 s
    def demo() -> None:
        gs = {
            "hero_cards": ["As", "Ks"],
            "board_cards": ["8c", "2h"],
            "pot": 42,
            "hero_stack": 109,
            "villain_stats": {"vpip": 0.24, "af": 2.3},
        }
        hud.state_updated.emit(gs, "RECOMMEND: RAISE to 9BB")

    QtCore.QTimer.singleShot(2000, demo)
    sys.exit(app.exec_())
