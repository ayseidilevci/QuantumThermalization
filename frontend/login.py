from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
)

from frontend.colors import Colors

class LoginWindow(QWidget):

    login_code_received = Signal(str)
    login_finished = Signal(int)

    login_successful = Signal()
    login_failed = Signal()

    def __init__(self, manager):
        super().__init__()

        self.manager = manager

        self.setWindowTitle("Quantum Inspire Experiment")

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Colors.BG_1};
                color: {Colors.TEXT};
            }}
        """)

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            60, 40, 60, 60
        )

        # -------------------------------------------------
        # Title
        # -------------------------------------------------

        self.title = QLabel(
            "Time to some fun with, QI Experiment"
        )

        self.title.setAlignment(Qt.AlignCenter)

        title_font = QFont()
        title_font.setPointSize(30)
        title_font.setBold(True)

        self.title.setFont(title_font)

        # -------------------------------------------------
        # Start button
        # -------------------------------------------------

        self.start_button = QPushButton(
            "START CONNECTION"
        )

        self.start_button.setFixedSize(
            280, 70
        )

        self.start_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.BG_2};
                color: {Colors.TEXT};
                border: none;
                border-radius: 12px;
                font-size: 18px;
                font-weight: bold;
            }}

            QPushButton:hover {{
                background-color: #293746;
            }}

            QPushButton:pressed {{
                background-color: #18222D;
            }}

            QPushButton:disabled {{
                background-color: #141D26;
                color: #59646E;
            }}
        """)

        # -------------------------------------------------
        # Login code
        # -------------------------------------------------

        self.code_label = QLabel("")

        self.code_label.setAlignment(
            Qt.AlignCenter
        )

        self.code_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT};
                font-size: 22px;
                font-weight: bold;
            }}
        """)

        self.code_label.hide()

        # -------------------------------------------------
        # Instructions
        # -------------------------------------------------

        self.check_label = QLabel(
            "Check if this matches with the one "
            "provided on the QI website"
        )

        self.check_label.setAlignment(
            Qt.AlignCenter
        )

        self.check_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT};
                font-size: 15px;
            }}
        """)

        self.check_label.hide()

        # -------------------------------------------------
        # Layout
        # -------------------------------------------------

        layout.addWidget(self.title)

        layout.addStretch()

        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)

        button_layout.addWidget(self.start_button)

        layout.addLayout(button_layout)

        layout.addSpacing(25)

        layout.addWidget(self.code_label)
        layout.addWidget(self.check_label)

        layout.addStretch()

        # -------------------------------------------------
        # Connections
        # -------------------------------------------------

        self.start_button.clicked.connect(
            self.start_login
        )

        self.login_code_received.connect(
            self.show_code
        )

        self.login_finished.connect(
            self.handle_login_finished
        )

    def start_login(self):

        self.start_button.setEnabled(False)

        self.manager.login(
            on_code=self.login_code_received.emit,
            on_complete=self.login_finished.emit,
        )

    def show_code(self, code):

        self.code_label.setText(
            f"This is your code: {code}"
        )

        self.code_label.show()
        self.check_label.show()

    def handle_login_finished(self, return_code):

        if return_code == 0:

            self.login_successful.emit()

        else:

            self.login_failed.emit()
            self.start_button.setEnabled(True)
