from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from frontend.colors import Colors

_SIDE_WIDTH = 280


class TopBar(QWidget):

    home_clicked = Signal()
    back_clicked = Signal()   # Run screen -> Setup

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

        button_style = f"""
            QPushButton {{
                background: rgba(255,255,255,0.12); color: {Colors.TEXT};
                border: none; border-radius: 8px; padding: 6px 14px; font-weight: 600;
            }}
            QPushButton:hover {{ background: rgba(255,255,255,0.22); }}
        """
        self.home_btn = QPushButton("🏠  Home")
        self.home_btn.setCursor(Qt.PointingHandCursor)
        self.home_btn.setStyleSheet(button_style)
        self.home_btn.clicked.connect(self.home_clicked.emit)
        self.home_btn.hide()  # hidden on the Login screen

        self.back_btn = QPushButton("←  Back to Setup")
        self.back_btn.setCursor(Qt.PointingHandCursor)
        self.back_btn.setStyleSheet(button_style)
        self.back_btn.clicked.connect(self.back_clicked.emit)
        self.back_btn.hide()  # Run screen only

        # Fixed-width sides keep the title centred whichever buttons show.
        left = QWidget()
        left.setStyleSheet("background: transparent;")
        left.setFixedWidth(_SIDE_WIDTH)
        left_layout = QHBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(self.home_btn)
        left_layout.addWidget(self.back_btn)
        left_layout.addStretch()
        layout.addWidget(left)

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

        spacer = QLabel("")
        spacer.setFixedWidth(_SIDE_WIDTH)
        layout.addWidget(spacer)

    def set_home_visible(self, visible: bool):
        self.home_btn.setVisible(visible)

    def set_back_visible(self, visible: bool):
        self.back_btn.setVisible(visible)
