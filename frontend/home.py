from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
)

from frontend.colors import Colors

class HomeWindow(QWidget):

    experiment_requested = Signal()

    def __init__(self, manager):
        super().__init__()

        self.manager = manager        

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            70, 60, 70, 60
        )

        # -------------------------------------------------
        # Cards
        # -------------------------------------------------

        cards_layout = QHBoxLayout()

        cards_layout.setSpacing(35)

        self.run_button = self.create_card(
            "RUN\nEXPERIMENT"
        )

        self.upload_button = self.create_card(
            "UPLOAD\nEXPERIMENT"
        )

        self.tutorial_button = self.create_card(
            "TUTORIAL"
        )

        cards_layout.addWidget(
            self.run_button
        )

        cards_layout.addWidget(
            self.upload_button
        )

        cards_layout.addWidget(
            self.tutorial_button
        )

        # -------------------------------------------------
        # Center cards
        # -------------------------------------------------

        layout.addStretch()

        layout.addLayout(
            cards_layout,
            stretch=3
        )

        layout.addStretch()

        # -------------------------------------------------
        # Connections
        # -------------------------------------------------

        self.upload_button.clicked.connect(
            self.open_upload
        )

        self.tutorial_button.clicked.connect(
            self.open_tutorial
        )

        self.run_button.clicked.connect(
            self.experiment_requested.emit
        )

    # -----------------------------------------------------
    # Card factory
    # -----------------------------------------------------

    def create_card(self, text):

        button = QPushButton(text)

        button.setMinimumSize(
            280,
            280
        )

        button.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        font = QFont()
        font.setPointSize(20)
        font.setBold(True)

        button.setFont(font)

        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.BG_2};
                color: {Colors.TEXT};

                border: 1px solid #2A3949;
                border-radius: 24px;

                padding: 30px;
            }}

            QPushButton:hover {{
                background: qlineargradient(
                    x1: 0, y1: 0,
                    x2: 1, y2: 1,
                    stop: 0 {Colors.GRADIENT_1_START},
                    stop: 1 {Colors.GRADIENT_1_END}
                );

                color: {Colors.TEXT};
            }}

            QPushButton:pressed {{
                background-color: #162330;
            }}
        """)

        return button

    def open_upload(self):
        pass

    def open_tutorial(self):
        pass
