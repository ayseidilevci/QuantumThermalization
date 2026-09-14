from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QHBoxLayout, QWidget

from frontend.colors import Colors


class TopBar(QWidget):

    def __init__(self):
        super().__init__()

        self.setFixedHeight(65)

        self.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(
                    x1: 0, y1: 0,
                    x2: 1, y2: 0,
                    stop: 0 {Colors.GRADIENT_2_START},
                    stop: 1 {Colors.GRADIENT_2_END}
                );
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(25, 0, 25, 0)

        title = QLabel("Quantum Inspire Experiment")

        font = QFont()
        font.setPointSize(20)
        font.setBold(True)

        title.setFont(font)
        title.setStyleSheet(f"""
            QLabel {{
                background: transparent;
                color: {Colors.TEXT};
            }}
        """)

        layout.addWidget(
            title,
            alignment=Qt.AlignCenter
        )
