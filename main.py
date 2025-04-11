import sys
from PyQt5.QtWidgets import QApplication

import MainWindowFile

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindowFile.MainWindow()
    window.show()
    sys.exit(app.exec_())