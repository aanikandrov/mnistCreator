import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFrame, QLineEdit,
                             QLabel, QSpacerItem, QSizePolicy, QMessageBox,
                             QProgressBar, QFileDialog, QSpinBox)
from PyQt5.QtCore import Qt, QRect, QPoint, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QMouseEvent, QPen, QFont, QIntValidator

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os




import MainWindowFile

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindowFile.MainWindow()
    window.show()
    sys.exit(app.exec_())