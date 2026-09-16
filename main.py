import sys

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QStackedWidget,
    QMessageBox,
)

from backend.qi_manager import QuantumInspireManager

from frontend.colors import Colors, Fonts
from frontend.topbar import TopBar
from frontend.login import LoginWindow
from frontend.home import HomeWindow
from frontend.backend_select import BackendSelectWindow
from frontend.setup import SetupWindow
from frontend.experiment import ExperimentWindow


def run():

    app = QApplication(sys.argv)

    # Global application style
    app.setStyleSheet(f"""
        QWidget {{
            background-color: {Colors.BG_1};
            color: {Colors.TEXT};
            font-family: {Fonts.FAMILY};
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
    backend_select = BackendSelectWindow(manager)
    setup = SetupWindow(manager)
    experiment = ExperimentWindow(manager)

    stack.addWidget(login)
    stack.addWidget(home)
    stack.addWidget(backend_select)
    stack.addWidget(setup)
    stack.addWidget(experiment)

    # Set by open_setup_from_home() while a Start/Tutorial click waits on
    # BackendSelectWindow's card pick — Load already carries a backend_name
    # and skips that screen, so this only matters for the other two paths.
    pending_tutorial = [False]

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

    def show_backend_select():
        topbar.set_home_visible(True)
        stack.setCurrentWidget(backend_select)

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
        if config is not None:
            # Load: the saved session already names a backend, so skip
            # straight past card selection.
            setup.set_backend(config.backend_name)
            setup.load_config(config)
            setup.set_tutorial_visible(show_tutorial)
            show_setup()
            return
        # Start / Tutorial: pick a backend first.
        pending_tutorial[0] = show_tutorial
        show_backend_select()

    def start_setup_from_backend_select(name, qi_backend):
        setup.set_backend(name, qi_backend)
        setup.load_config(None)
        setup.set_tutorial_visible(pending_tutorial[0])
        show_setup()

    def start_run_from_setup(config):
        experiment.start_run(config)
        show_run()

    # Login -> Home
    login.login_successful.connect(show_home)

    # Home -> Backend Select (Start / Tutorial) or straight to Setup (Load)
    home.open_setup.connect(open_setup_from_home)

    # Backend Select -> Setup
    backend_select.backend_chosen.connect(start_setup_from_backend_select)

    # Setup -> Backend Select / Run
    setup.change_backend.connect(show_backend_select)
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
