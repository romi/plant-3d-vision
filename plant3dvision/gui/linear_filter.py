#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QDoubleSpinBox, QPushButton,
                             QFileDialog)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from plant3dvision.proc2d import linear


class RGBFilterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RGB Linear Filter and Threshold")
        self.setGeometry(100, 100, 800, 600)

        # Image placeholders
        self.original_img = None
        self.filtered_img = None
        self.mask = None

        # Initialize UI
        self.initUI()

    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Controls panel
        controls_layout = QHBoxLayout()

        # RGB sliders
        sliders_layout = QVBoxLayout()

        # Red slider
        red_layout = QHBoxLayout()
        red_label = QLabel("Red:")
        self.red_slider = QSlider(Qt.Horizontal)
        self.red_slider.setRange(0, 100)
        self.red_slider.setValue(50)
        self.red_value = QLabel("0.5")
        red_layout.addWidget(red_label)
        red_layout.addWidget(self.red_slider)
        red_layout.addWidget(self.red_value)
        sliders_layout.addLayout(red_layout)

        # Green slider
        green_layout = QHBoxLayout()
        green_label = QLabel("Green:")
        self.green_slider = QSlider(Qt.Horizontal)
        self.green_slider.setRange(0, 100)
        self.green_slider.setValue(100)
        self.green_value = QLabel("1.0")
        green_layout.addWidget(green_label)
        green_layout.addWidget(self.green_slider)
        green_layout.addWidget(self.green_value)
        sliders_layout.addLayout(green_layout)

        # Blue slider
        blue_layout = QHBoxLayout()
        blue_label = QLabel("Blue:")
        self.blue_slider = QSlider(Qt.Horizontal)
        self.blue_slider.setRange(0, 100)
        self.blue_slider.setValue(50)
        self.blue_value = QLabel("0.5")
        blue_layout.addWidget(blue_label)
        blue_layout.addWidget(self.blue_slider)
        blue_layout.addWidget(self.blue_value)
        sliders_layout.addLayout(blue_layout)

        # Threshold control
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.0, 1.0)
        self.threshold_spinbox.setSingleStep(0.01)
        self.threshold_spinbox.setValue(0.3)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_spinbox)
        sliders_layout.addLayout(threshold_layout)

        # Load and process buttons
        buttons_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Image")
        self.process_button = QPushButton("Process")
        self.process_button.setEnabled(False)
        buttons_layout.addWidget(self.load_button)
        buttons_layout.addWidget(self.process_button)
        sliders_layout.addLayout(buttons_layout)

        # Add sliders to controls
        controls_layout.addLayout(sliders_layout)

        # Add controls to main layout
        main_layout.addLayout(controls_layout)

        # Image display area
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        # Connect signals
        self.red_slider.valueChanged.connect(self.update_red_value)
        self.green_slider.valueChanged.connect(self.update_green_value)
        self.blue_slider.valueChanged.connect(self.update_blue_value)
        self.load_button.clicked.connect(self.load_image)
        self.process_button.clicked.connect(self.process_image)

    def update_red_value(self, value):
        self.red_value.setText(f"{value / 100:.2f}")

    def update_green_value(self, value):
        self.green_value.setText(f"{value / 100:.2f}")

    def update_blue_value(self, value):
        self.blue_value.setText(f"{value / 100:.2f}")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )

        if file_path:
            # Load the image
            try:
                from PIL import Image
                img = np.array(Image.open(file_path).convert('RGB')) / 255.0
                self.original_img = img

                # Display the original image
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.imshow(self.original_img)
                ax.set_title("Original Image")
                ax.axis('off')
                self.canvas.draw()

                # Enable process button
                self.process_button.setEnabled(True)
            except Exception as e:
                print(f"Error loading image: {e}")

    def process_image(self):
        if self.original_img is None:
            return

        # Get values from sliders
        red_coef = self.red_slider.value() / 100.0
        green_coef = self.green_slider.value() / 100.0
        blue_coef = self.blue_slider.value() / 100.0
        threshold = self.threshold_spinbox.value()

        # Apply linear filter
        rgb_coefficients = [red_coef, green_coef, blue_coef]
        self.filtered_img = linear(self.original_img, rgb_coefficients)

        # Apply threshold
        self.mask = self.filtered_img > threshold

        # Display results
        self.figure.clear()

        # Original image
        ax1 = self.figure.add_subplot(131)
        ax1.imshow(self.original_img)
        ax1.set_title("Original")
        ax1.axis('off')

        # Filtered image
        ax2 = self.figure.add_subplot(132)
        ax2.imshow(self.filtered_img, cmap='gray')
        ax2.set_title(f"Filtered [R:{red_coef:.2f}, G:{green_coef:.2f}, B:{blue_coef:.2f}]")
        ax2.axis('off')

        # Mask
        ax3 = self.figure.add_subplot(133)
        ax3.imshow(self.mask, cmap='binary')
        ax3.set_title(f"Mask (threshold: {threshold:.2f})")
        ax3.axis('off')

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RGBFilterApp()
    window.show()
    sys.exit(app.exec_())