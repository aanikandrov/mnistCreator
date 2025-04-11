import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PyQt5.QtWidgets import (QFrame, QLabelx)
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QMouseEvent, QPen, QFont

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class DrawingArea(QFrame):
    layer_selected = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setMinimumSize(400, 400)
        self.squares = []
        self.layer_types = []
        self.dragged_square_index = -1
        self.drag_offset = None
        self.counter_label = QLabel("Слоёв: 0")
        self.counter_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.counter_label.setStyleSheet("font-size: 12px; color: gray;")


    def add_square(self, color, text="", layer_type=""):
        size = 70
        padding = 15

        if not self.squares:
            x, y = padding, padding
        else:
            x, y = padding, padding
            found = False

            while not found:
                new_rect = QRect(x, y, size, size)
                collision = False

                for square in self.squares:
                    existing_rect = QRect(square['x'], square['y'], square['size'], square['size'])
                    if new_rect.intersects(existing_rect):
                        collision = True
                        break

                if not collision:
                    found = True
                else:
                    x += size + padding
                    if x + size > self.width() - padding:
                        x = padding
                        y += size + padding

        self.squares.append({
            'x': x,
            'y': y,
            'size': size,
            'color': color,
            'center': QPoint(x + size // 2, y + size // 2),
            'text': text
        })
        self.layer_types.append(layer_type)
        self.update_counter()
        self.update()

    def remove_last_square(self):
        if self.squares:
            self.squares.pop()
            self.layer_types.pop()
            self.update_counter()
            self.update()

    def update_counter(self):
        self.counter_label.setText(f"Слоёв: {len(self.squares)}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.black, 1, Qt.DashLine))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

        if len(self.squares) > 1:
            pen = QPen(QColor(100, 200, 100), 6)
            painter.setPen(pen)

            for i in range(1, len(self.squares)):
                start = self.squares[i - 1]['center']
                end = self.squares[i]['center']
                painter.drawLine(start, end)

        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        for square in self.squares:
            painter.fillRect(
                square['x'],
                square['y'],
                square['size'],
                square['size'],
                square['color']
            )

            painter.setPen(Qt.black)
            text_rect = QRect(
                square['x'],
                square['y'],
                square['size'],
                square['size']
            )
            painter.drawText(text_rect, Qt.AlignCenter, square['text'])

        painter.end()
        self.counter_label.setGeometry(
            self.width() - 120,
            self.height() - 30,
            110,
            20
        )
        self.counter_label.setParent(self)
        self.counter_label.show()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            for i, square in enumerate(self.squares):
                square_rect = QRect(
                    square['x'],
                    square['y'],
                    square['size'],
                    square['size']
                )
                if square_rect.contains(pos):
                    self.layer_selected.emit(i)
                    self.dragged_square_index = i
                    self.drag_offset = pos - square_rect.topLeft()
                    break

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragged_square_index != -1 and event.buttons() & Qt.LeftButton:
            pos = event.pos()
            square = self.squares[self.dragged_square_index]
            size = square['size']

            new_x = pos.x() - self.drag_offset.x()
            new_y = pos.y() - self.drag_offset.y()

            new_x = max(0, min(new_x, self.width() - size))
            new_y = max(0, min(new_y, self.height() - size))

            new_rect = QRect(new_x, new_y, size, size)
            collision = False

            for i, other_square in enumerate(self.squares):
                if i == self.dragged_square_index:
                    continue
                other_rect = QRect(
                    other_square['x'],
                    other_square['y'],
                    other_square['size'],
                    other_square['size']
                )
                if new_rect.intersects(other_rect):
                    collision = True
                    break

            if not collision:
                square['x'] = new_x
                square['y'] = new_y
                square['center'] = QPoint(new_x + size // 2, new_y + size // 2)
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.dragged_square_index = -1
            self.drag_offset = None
