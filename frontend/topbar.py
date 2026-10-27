from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from frontend.colors import Colors


class TopBar(QWidget):

    home_clicked = Signal()

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

        self.home_btn = QPushButton("🏠  Home")
        self.home_btn.setCursor(Qt.PointingHandCursor)
        self.home_btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(255,255,255,0.12); color: {Colors.TEXT};
                border: none; border-radius: 8px; padding: 6px 14px; font-weight: 600;
            }}
            QPushButton:hover {{ background: rgba(255,255,255,0.22); }}
        """)
        self.home_btn.clicked.connect(self.home_clicked.emit)
        self.home_btn.hide()  # hidden on the Login screen
        layout.addWidget(self.home_btn)

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

        layout.addWidget(title, stretch=1, alignment=Qt.AlignCenter)

        # balances the Home button's width so the title stays visually
        # centered whether or not the button is shown
        spacer = QLabel("")
        spacer.setFixedWidth(90)
        layout.addWidget(spacer)

    def set_home_visible(self, visible: bool):
        self.home_btn.setVisible(visible)
