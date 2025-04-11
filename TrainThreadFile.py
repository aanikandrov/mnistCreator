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




class TrainThread(QThread):
    update_signal = pyqtSignal(int, float, float)  # epoch, loss, accuracy
    finished_signal = pyqtSignal(dict)  # history data

    def __init__(self, model, train_loader, test_loader, epochs_count):
        super().__init__()
        self.model = model
        self.epochs_count = epochs_count
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.history = {'epochs': [], 'loss': [], 'accuracy': []}

    def run(self):
        # Обучение
        for epoch in range(1, self.epochs_count + 1):
            self.model.train()
            running_loss = 0.0
            total = 0
            correct = 0

            for i, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                if i % 100 == 0:
                    accuracy = 100 * correct / total
                    self.update_signal.emit(epoch, running_loss / 100, accuracy)
                    running_loss = 0.0
                    correct = 0
                    total = 0

            # Сохраняем метрики после каждой эпохи
            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = 100 * correct / total
            self.history['epochs'].append(epoch)
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(epoch_accuracy)

        # Финальное тестирование
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_accuracy = 100 * correct / total
        self.history['test_accuracy'] = test_accuracy

        torch.save(self.model.state_dict(), 'modelMNIST.pth')
        # torch.save(self.model, 'modelMNIST.pth')
        self.finished_signal.emit(self.history)
