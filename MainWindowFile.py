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

import ResultsWindowFile
import TrainThreadFile
import DrawingAreaFile

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.setWindowTitle("Neural Network Builder for MNIST")
        self.setGeometry(100, 100, 800, 600)
        self.epochs = 5

        self.model = None
        self.train_loader = None
        self.test_loader = None

        # Инициализация UI
        self.init_ui()

        # Загрузка данных MNIST
        self.load_mnist_data()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)



        # Область рисования
        self.drawing_area = DrawingAreaFile.DrawingArea()
        main_layout.addWidget(self.drawing_area, stretch=1)

        # Правая панель
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Верхние кнопки
        top_buttons = QWidget()
        top_buttons_layout = QVBoxLayout(top_buttons)

        epochs_widget = QWidget()
        epochs_layout = QHBoxLayout(epochs_widget)
        epochs_label = QLabel("Epochs:")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(5)
        epochs_layout.addWidget(epochs_label)
        epochs_layout.addWidget(self.epochs_spin)
        top_buttons_layout.addWidget(epochs_widget)

        # Кнопка Linear
        red_button = QPushButton("Linear Layer")
        red_button.setStyleSheet("""
            background-color: #ff9999;
            color: black;
            padding: 8px;
        """)
        red_button.clicked.connect(self.add_red_square)
        top_buttons_layout.addWidget(red_button)

        # Кнопка Conv2d
        blue_button = QPushButton("Conv2d Layer")
        blue_button.setStyleSheet("""
            background-color: #9999ff;
            color: black;
            padding: 8px;
        """)
        blue_button.clicked.connect(self.add_blue_square)
        top_buttons_layout.addWidget(blue_button)

        # Поле ввода и кнопка удаления
        back_group = QWidget()
        back_layout = QVBoxLayout(back_group)

        self.number_input = QLineEdit()
        self.number_input.setPlaceholderText("Enter units/kernels")
        self.number_input.setValidator(QIntValidator(1, 500))
        back_layout.addWidget(self.number_input)

        back_button = QPushButton("Remove Last Layer")
        back_button.setStyleSheet("""
            background-color: #cccccc;
            color: black;
            padding: 8px;
        """)
        back_button.clicked.connect(self.drawing_area.remove_last_square)
        back_layout.addWidget(back_button)

        top_buttons_layout.addWidget(back_group)
        top_buttons_layout.addStretch()

        right_layout.addWidget(top_buttons)

        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.epochs)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Эпоха: %v")
        right_layout.addWidget(self.progress_bar)

        # Нижние кнопки
        bottom_buttons = QWidget()
        bottom_layout = QHBoxLayout(bottom_buttons)
        bottom_layout.addStretch()

        open_button = QPushButton("Open Model")
        open_button.setStyleSheet("""
                    background-color: #aaaaff;
                    color: black;
                    padding: 8px 16px;
                """)
        open_button.clicked.connect(self.open_model)
        bottom_layout.addWidget(open_button)

        self.train_button = QPushButton("Train & Test")
        self.train_button.setStyleSheet("""
            background-color: #aaffaa;
            color: black;
            padding: 8px 16px;
        """)
        self.train_button.clicked.connect(self.train_and_test_model)
        bottom_layout.addWidget(self.train_button)

        right_layout.addWidget(bottom_buttons)

        main_layout.addWidget(right_panel)

    def open_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Model", "", "PyTorch Model Files (*.pth)")
        if not file_path:
            return

        try:
            model = torch.load(file_path)
            if not isinstance(model, MainWindow.DynamicModel):
                raise ValueError("Invalid model type")

            self.drawing_area.squares.clear()
            self.drawing_area.layer_types.clear()

            # Пропускаем последний Linear слой (классификатор)
            for layer in model.layers[:-1]:
                if isinstance(layer, nn.Conv2d):
                    self.drawing_area.add_square(
                        QColor(150, 150, 200),
                        str(layer.out_channels),
                        "Conv2d"
                    )
                elif isinstance(layer, nn.Linear):
                    self.drawing_area.add_square(
                        QColor(200, 150, 150),
                        str(layer.out_features),
                        "Linear"
                    )

            self.drawing_area.update()
            QMessageBox.information(self, "Success", "Model loaded successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load model: {str(e)}")

    def add_red_square(self):
        text = self.number_input.text() if self.number_input.text() else "0"
        self.drawing_area.add_square(QColor(200, 150, 150), text, "Linear")

    def add_blue_square(self):
        text = self.number_input.text() if self.number_input.text() else "0"
        self.drawing_area.add_square(QColor(150, 150, 200), text, "Conv2d")

    def load_mnist_data(self):
        # Преобразования для данных MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Загрузка тренировочных и тестовых данных
        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        # Создание DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1000,
            shuffle=False
        )

    def create_model_from_layers(self):
        class DynamicModel(nn.Module):
            def __init__(self, layer_types, squares):
                super(DynamicModel, self).__init__()
                self.layers = nn.ModuleList()

                in_channels = 1  # Для MNIST (1 канал)
                input_size = 28  # Размер изображения MNIST
                has_flattened = False

                for i, (layer_type, square) in enumerate(zip(layer_types, squares)):
                    text = square['text']
                    if layer_type == "Conv2d":
                        if has_flattened:
                            raise ValueError("Conv2d нельзя добавлять после Linear слоя!")

                        out_channels = int(text) if text else 32
                        self.layers.append(
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
                        in_channels = out_channels
                        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                        self.layers.append(nn.ReLU())
                        input_size = input_size // 2

                    elif layer_type == "Linear":
                        if not has_flattened:
                            # Добавляем Flatten только перед первым Linear
                            self.layers.append(nn.Flatten())
                            flat_size = in_channels * (input_size ** 2)
                            has_flattened = True
                        else:
                            flat_size = in_channels

                        out_features = int(text) if text else 128
                        self.layers.append(nn.Linear(flat_size, out_features))
                        self.layers.append(nn.ReLU())
                        in_channels = out_features

                # Финальный классификатор
                if has_flattened:
                    self.layers.append(nn.Linear(in_channels, 10))
                else:
                    # Если нет Linear слоев, добавляем их автоматически
                    self.layers.append(nn.Flatten())
                    flat_size = in_channels * (input_size ** 2)
                    self.layers.append(nn.Linear(flat_size, 128))
                    self.layers.append(nn.ReLU())
                    self.layers.append(nn.Linear(128, 10))

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = DynamicModel(self.drawing_area.layer_types, self.drawing_area.squares)
        print(model)
        return model

    def train_and_test_model(self):
        if not self.drawing_area.layer_types:
            QMessageBox.warning(self, "Warning", "No layers added to the model!")
            return

        self.epochs = self.epochs_spin.value()
        self.model = self.create_model_from_layers()

        # Создаем и запускаем поток обучения
        self.train_thread = TrainThreadFile.TrainThread(self.model, self.train_loader, self.test_loader, self.epochs)
        self.train_thread.update_signal.connect(self.update_progress)
        self.train_thread.finished_signal.connect(self.show_results)
        self.train_thread.start()

        # Блокируем кнопку на время обучения
        self.train_button.setEnabled(False)
        self.train_button.setText("Training...")

    def update_progress(self, epoch, loss):
        self.progress_bar.setValue(epoch)
        self.statusBar().showMessage(f"Эпоха: {epoch}, Loss: {loss:.4f}")

    def show_results(self, history):
        self.train_button.setEnabled(True)
        self.train_button.setText("Train & Test")

        # Показываем результаты обучения
        msg = QMessageBox()
        msg.setWindowTitle("Training Complete")
        msg.setText(
            f"Модель успешно обучена!\nТочность на тестовом наборе: {history['test_accuracy']:.2f}%\nМодель сохранена как modelMNIST.pth")
        msg.exec_()

        # Показываем окно с графиками
        self.results_window = ResultsWindowFile.TrainingResultsWindow(history)
        self.results_window.show()