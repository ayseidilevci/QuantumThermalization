import sys

from PySide6.QtWidgets import QApplication, QStackedWidget

from backend.qi_manager import QuantumInspireManager
from frontend.login import LoginWindow
from frontend.home import HomeWindow


def run():

    app = QApplication(sys.argv)

    manager = QuantumInspireManager()

    window = QStackedWidget()

    login = LoginWindow(manager)
    home = HomeWindow()

    window.addWidget(login)
    window.addWidget(home)

    login.login_successful.connect(
        lambda: window.setCurrentWidget(home)
    )

    if manager.check_login():
        window.setCurrentWidget(home)
    else:
        window.setCurrentWidget(login)

    window.resize(1100, 700)
    window.show()

    # Center window
    screen = window.screen().availableGeometry()

    window.move(
        screen.center() - window.rect().center()
    )

    sys.exit(app.exec())


if __name__ == "__main__":
    run()
