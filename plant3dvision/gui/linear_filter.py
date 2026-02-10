#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QDoubleSpinBox
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QSlider
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget
from PIL import Image

# Switched to backend_qtagg for Qt6 compatibility (PySide6)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from plant3dvision.proc2d import linear


class RGBFilterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linear Filter and Threshold")
        self.setGeometry(100, 100, 800, 600)

        # Image placeholders
        self.source_image: Optional[Image] = None  # PIL source image
        self.original_img = None
        self.filtered_img = None
        self.mask = None

        self.ch1_value = None
        self.ch2_value = None
        self.ch3_value = None

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

        # Sliders sliders
        sliders_layout = QVBoxLayout()

        # Color Space Selector
        cs_layout = QHBoxLayout()
        cs_label = QLabel("Color Space:")
        self.color_space_combo = QComboBox()
        self.color_space_combo.addItems(["RGB", "HSV", "YCbCr"])
        self.color_space_combo.currentTextChanged.connect(self.update_channel_labels)
        cs_layout.addWidget(cs_label)
        cs_layout.addWidget(self.color_space_combo)
        sliders_layout.addLayout(cs_layout)

        # Channel 1 slider
        ch1_layout = QHBoxLayout()
        self.ch1_label = QLabel("Red:")
        self.ch1_slider = QSlider()
        self.ch1_slider.setOrientation(Qt.Orientation.Horizontal)
        self.ch1_slider.setRange(0, 100)
        self.ch1_slider.setValue(50)
        self.ch1_slider.setMinimumWidth(200)
        self.ch1_value = QLabel("0.5")
        ch1_layout.addWidget(self.ch1_label)
        ch1_layout.addWidget(self.ch1_slider)
        ch1_layout.addWidget(self.ch1_value)
        sliders_layout.addLayout(ch1_layout)

        # Channel 2 slider
        ch2_layout = QHBoxLayout()
        self.ch2_label = QLabel("Green:")
        self.ch2_slider = QSlider()
        self.ch2_slider.setOrientation(Qt.Orientation.Horizontal)
        self.ch2_slider.setRange(0, 100)
        self.ch2_slider.setValue(100)
        self.ch2_slider.setMinimumWidth(200)
        self.ch2_value = QLabel("1.0")
        ch2_layout.addWidget(self.ch2_label)
        ch2_layout.addWidget(self.ch2_slider)
        ch2_layout.addWidget(self.ch2_value)
        sliders_layout.addLayout(ch2_layout)

        # Channel 3 slider
        ch3_layout = QHBoxLayout()
        self.ch3_label = QLabel("Blue:")
        self.ch3_slider = QSlider()
        self.ch3_slider.setOrientation(Qt.Orientation.Horizontal)
        self.ch3_slider.setRange(0, 100)
        self.ch3_slider.setValue(50)
        self.ch3_slider.setMinimumWidth(200)
        self.ch3_value = QLabel("0.5")
        ch3_layout.addWidget(self.ch3_label)
        ch3_layout.addWidget(self.ch3_slider)
        ch3_layout.addWidget(self.ch3_value)
        sliders_layout.addLayout(ch3_layout)

        # Threshold control
        threshold_layout = QHBoxLayout()

        min_thresh_label = QLabel("Min Threshold:")
        self.min_threshold_spinbox = QDoubleSpinBox()
        self.min_threshold_spinbox.setRange(0.0, 1.0)
        self.min_threshold_spinbox.setSingleStep(0.01)
        self.min_threshold_spinbox.setValue(0.3)

        max_thresh_label = QLabel("Max Threshold:")
        self.max_threshold_spinbox = QDoubleSpinBox()
        self.max_threshold_spinbox.setRange(0.0, 1.0)
        self.max_threshold_spinbox.setSingleStep(0.01)
        self.max_threshold_spinbox.setValue(1.0)

        threshold_layout.addWidget(min_thresh_label)
        threshold_layout.addWidget(self.min_threshold_spinbox)
        threshold_layout.addWidget(max_thresh_label)
        threshold_layout.addWidget(self.max_threshold_spinbox)
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
        self.ch1_slider.valueChanged.connect(self.update_ch1_value)
        self.ch2_slider.valueChanged.connect(self.update_ch2_value)
        self.ch3_slider.valueChanged.connect(self.update_ch3_value)
        self.load_button.clicked.connect(self.load_image)
        self.process_button.clicked.connect(self.process_image)

    def update_channel_labels(self, mode):
        labels = {
            "RGB": ("Red:", "Green:", "Blue:"),
            "HSV": ("Hue:", "Saturation:", "Value:"),
            "YCbCr": ("Luminance (Y):", "Blue Diff (Cb):", "Red Diff (Cr):")
        }
        l1, l2, l3 = labels.get(mode, ("Ch1:", "Ch2:", "Ch3:"))
        self.ch1_label.setText(l1)
        self.ch2_label.setText(l2)
        self.ch3_label.setText(l3)

    def update_ch1_value(self, value):
        self.ch1_value.setText(f"{value / 100:.2f}")

    def update_ch2_value(self, value):
        self.ch2_value.setText(f"{value / 100:.2f}")

    def update_ch3_value(self, value):
        self.ch3_value.setText(f"{value / 100:.2f}")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )

        if file_path:
            # Load the image
            try:
                self.source_image = Image.open(file_path).convert("RGB")
                img = np.array(self.source_image) / 255.0
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
        c1_coef = self.ch1_slider.value() / 100.0
        c2_coef = self.ch2_slider.value() / 100.0
        c3_coef = self.ch3_slider.value() / 100.0
        min_threshold = self.min_threshold_spinbox.value()
        max_threshold = self.max_threshold_spinbox.value()
        mode = self.color_space_combo.currentText()

        # Prepare image in selected color space
        if mode == 'RGB':
            img_to_filter = self.original_img
        else:
            # Convert using PIL and normalize to [0, 1]
            converted = self.source_image.convert(mode)
            img_to_filter = np.array(converted) / 255.0

        # Apply linear filter
        coefficients = [c1_coef, c2_coef, c3_coef]
        self.filtered_img = linear(self.original_img, coefficients, colorspace=mode)

        # Apply threshold
        self.mask = (self.filtered_img >= min_threshold) & (self.filtered_img <= max_threshold)

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
        ax2.set_title(f"Filtered [{mode}]\nC1:{c1_coef:.2f}, C2:{c2_coef:.2f}, C3:{c3_coef:.2f}")
        ax2.axis('off')

        # Mask
        ax3 = self.figure.add_subplot(133)
        ax3.imshow(self.mask, cmap='binary')
        ax3.set_title(f"Mask ({min_threshold:.2f} <= v <= {max_threshold:.2f})")
        ax3.axis('off')

        self.figure.tight_layout()
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    window = RGBFilterApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
