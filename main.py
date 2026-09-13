import sys

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QStackedWidget,
)

from backend.qi_manager import QuantumInspireManager

from frontend.colors import Colors
from frontend.topbar import TopBar
from frontend.login import LoginWindow
from frontend.home import HomeWindow
from frontend.experiment import ExperimentWindow

def run():

    app = QApplication(sys.argv)

    # Global application style
    app.setStyleSheet(f"""
        QWidget {{
            background-color: {Colors.BG_1};
            color: {Colors.TEXT};
        }}
    """)

    # Backend
    manager = QuantumInspireManager()

    # Main window
    window = QWidget()

    window.resize(1100, 700)

    layout = QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    topbar = TopBar()

    layout.addWidget(topbar)

    stack = QStackedWidget()

    login = LoginWindow(manager)
    home = HomeWindow(manager)
    experiment = ExperimentWindow(manager)

    stack.addWidget(login)
    stack.addWidget(home)
    stack.addWidget(experiment)

    layout.addWidget(stack)

    login.login_successful.connect(
        lambda: stack.setCurrentWidget(home)
    )

    # Home -> Experiment
    home.experiment_requested.connect(
        lambda: stack.setCurrentWidget(experiment)
    )

    # Experiment -> Home
    experiment.go_home.connect(
        lambda: stack.setCurrentWidget(home)
    )


    if manager.check_login():
        stack.setCurrentWidget(home)
    else:
        stack.setCurrentWidget(login)

    window.show()

    screen = window.screen().availableGeometry()

    window.move(
        screen.center() - window.rect().center()
    )

    sys.exit(app.exec())


if __name__ == "__main__":
    run()
