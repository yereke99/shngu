# run.py
import sys
from PyQt5.QtWidgets import QApplication
from main import PumpDiagApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PumpDiagApp()
    window.show()
    sys.exit(app.exec_())