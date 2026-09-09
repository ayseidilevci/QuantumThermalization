from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
)


class HomeWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Quantum Inspire Experiment")

        layout = QVBoxLayout()

        self.run_button = QPushButton("Run experiment")
        self.upload_button = QPushButton("Upload experiment")
        self.tutorial_button = QPushButton("Tutorial")

        layout.addWidget(self.run_button)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.tutorial_button)

        self.setLayout(layout)

        self.run_button.clicked.connect(self.open_experiment)
        self.upload_button.clicked.connect(self.open_upload)
        self.tutorial_button.clicked.connect(self.open_tutorial)

    def open_experiment(self):
        pass

    def open_upload(self):
        pass

    def open_tutorial(self):
        pass
