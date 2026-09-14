import sys

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QStackedWidget,
    QMessageBox,
)

from backend.qi_manager import QuantumInspireManager

from frontend.colors import Colors
from frontend.topbar import TopBar
from frontend.login import LoginWindow
from frontend.home import HomeWindow
from frontend.setup import SetupWindow
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

    window.resize(1200, 800)

    layout = QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    topbar = TopBar()

    layout.addWidget(topbar)

    stack = QStackedWidget()

    login = LoginWindow(manager)
    home = HomeWindow(manager)
    setup = SetupWindow(manager)
    experiment = ExperimentWindow(manager)

    stack.addWidget(login)
    stack.addWidget(home)
    stack.addWidget(setup)
    stack.addWidget(experiment)

    layout.addWidget(stack)

    # -------------------------------------------------
    # Navigation
    # -------------------------------------------------

    def show_login():
        topbar.set_home_visible(False)
        stack.setCurrentWidget(login)

    def show_home():
        topbar.set_home_visible(True)
        stack.setCurrentWidget(home)

    def show_setup():
        topbar.set_home_visible(True)
        stack.setCurrentWidget(setup)

    def show_run():
        topbar.set_home_visible(True)
        stack.setCurrentWidget(experiment)

    def confirm_leave_run() -> bool:
        """If a run is live, ask before navigating away and stop it if
        confirmed. Returns True once it's safe to navigate."""
        if not experiment.is_running:
            return True
        reply = QMessageBox.question(
            window,
            "Stop experiment?",
            "An experiment is still running. Leaving this screen will stop it.\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            experiment.stop()
            return True
        return False

    def go_home_from_anywhere():
        if stack.currentWidget() is experiment and not confirm_leave_run():
            return
        show_home()

    def go_to_setup_from_run():
        if not confirm_leave_run():
            return
        show_setup()

    def open_setup_from_home(config, show_tutorial):
        setup.load_config(config)
        setup.set_tutorial_visible(show_tutorial)
        show_setup()

    def start_run_from_setup(config):
        experiment.start_run(config)
        show_run()

    # Login -> Home
    login.login_successful.connect(show_home)

    # Home -> Setup (Start / Load / Tutorial all land here)
    home.open_setup.connect(open_setup_from_home)

    # Setup -> Home / Run
    setup.go_home.connect(show_home)
    setup.run_requested.connect(start_run_from_setup)

    # Run -> Setup
    experiment.back_to_setup.connect(go_to_setup_from_run)

    # Persistent top bar -> Home, from anywhere (guarded while a run is live)
    topbar.home_clicked.connect(go_home_from_anywhere)

    if manager.check_login():
        show_home()
    else:
        show_login()

    window.show()

    screen = window.screen().availableGeometry()

    window.move(
        screen.center() - window.rect().center()
    )

    sys.exit(app.exec())


if __name__ == "__main__":
    run()
