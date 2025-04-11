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


class TrainingResultsWindow(QMainWindow):
    def __init__(self, history):
        super().__init__()
        self.setWindowTitle("Training Results")
        self.setGeometry(200, 200, 800, 600)

        self.history = history
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # График точности
        self.figure1 = plt.figure(figsize=(10, 4))
        self.canvas1 = FigureCanvas(self.figure1)
        ax1 = self.figure1.add_subplot(111)
        ax1.plot(self.history['epochs'], self.history['accuracy'], 'b-')
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        layout.addWidget(self.canvas1)

        # График потерь
        self.figure2 = plt.figure(figsize=(10, 4))
        self.canvas2 = FigureCanvas(self.figure2)
        ax2 = self.figure2.add_subplot(111)
        ax2.plot(self.history['epochs'], self.history['loss'], 'r-')
        ax2.set_title('Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        layout.addWidget(self.canvas2)

        # Кнопка закрытия
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
