#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Linear Filter GUI

A small graphical application that lets you load an image, apply a customizable linear combination of its color channels, and visualize the filtered result together with a binary threshold mask.
It is useful for quickly exploring channel‑mixing effects and extracting regions of interest based on intensity thresholds.

## Key Features

- **Interactive UI** built with PySide6 offering sliders for three channel coefficients (range 0‑1).
- **Multiple color‑space support**: RGB, HSV, and YCbCr can be selected on‑the‑fly.
- **Real‑time preview** of the original image, the filtered grayscale image, and the binary mask using Matplotlib.
- **Adjustable thresholding** with minimum and maximum spin boxes to create precise binary masks.
- **Command‑line entry point** via Click, allowing the app to start with a pre‑loaded image.

## Usage Examples

```shell
# Run the GUI without an image (you can load one later via the “Load Image” button)
linear_filter

# Start the GUI and preload an image
linear_filter path/to/your/photo.jpg
```

When the application launches, use the sliders to set the weighting of each channel, choose a color space from the dropdown, and adjust the threshold spin boxes.
Press **Process** to see the filtered image and the corresponding mask.
"""

import sys

import click
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QDoubleSpinBox
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QSlider
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget
# Switched to backend_qtagg for Qt6 compatibility (PySide6)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from plant3dvision.proc2d import linear


class RGBFilterApp(QMainWindow):
    """RGB linear filter GUI application.

    Provides an interactive interface to load an image, apply a linear
    combination of its color channels, and visualize the resulting filtered
    image and threshold mask.

    Attributes
    ----------
    source_image : Image.Image
        The original image loaded via :pymod:`PIL`, converted to ``RGB``.
    original_img : numpy.ndarray
        Normalized (``[0, 1]``) NumPy array of the original image.
    filtered_img : numpy.ndarray
        Image obtained after applying the linear filter.
    mask : numpy.ndarray
        Boolean mask where ``True`` indicates pixel values within the chosen
        threshold range.
    ch1_value, ch2_value, ch3_value : PySide6.QtWidgets.QLabel
        Labels that display the current scaling factors for the three channels.
    """

    def __init__(self):
        """Initialize the main window and UI elements.

        Set the window title, geometry and creates placeholders for image data and channel coefficients.
        Then call `initUI` to build the graphical user interface.
        """
        super().__init__()
        self.setWindowTitle("Linear Filter and Threshold")
        self.setGeometry(100, 100, 800, 600)

        # Image placeholders
        self.source_image: Image = None  # PIL source image
        self.original_img = None
        self.filtered_img = None
        self.mask = None

        self.ch1_value = None
        self.ch2_value = None
        self.ch3_value = None

        # Initialize UI
        self.initUI()

    def initUI(self):
        """Create and arrange all widgets of the GUI.

        The layout consists of three slider panels (one per channel), a color‑space selector, threshold spin boxes,
        and a Matplotlib canvas for image display. Signal/slot connections are also set up here.
        """
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Controls panel
        controls_layout = QHBoxLayout()

        # Sliders
        sliders_layout = QVBoxLayout()

        # Color Space Selector
        cs_layout = QHBoxLayout()
        cs_label = QLabel("Color Space:")
        self.color_space_combo = QComboBox()
        self.color_space_combo.addItems(["RGB", "HSV", "YCbCr"])
        self.color_space_combo.currentTextChanged.connect(self.update_channel_labels)

        # ? button to show help
        self.cs_help_button = QPushButton("?")
        self.cs_help_button.setFixedSize(24, 24)
        self.cs_help_button.setToolTip("Show information about color spaces and channel sliders")
        self.cs_help_button.clicked.connect(self.show_color_space_help)

        cs_layout.addWidget(cs_label)
        cs_layout.addWidget(self.color_space_combo)
        cs_layout.addWidget(self.cs_help_button)
        sliders_layout.addLayout(cs_layout)

        # Channel 1 slider
        ch1_layout = QHBoxLayout()
        self.ch1_label = QLabel("Red:")
        self.ch1_slider = QSlider()
        self.ch1_slider.setOrientation(Qt.Orientation.Horizontal)
        self.ch1_slider.setRange(0, 100)
        self.ch1_slider.setValue(50)
        self.ch1_slider.setMinimumWidth(200)
        # Show tick marks on the channel slider
        self.ch1_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ch1_slider.setTickInterval(10)
        self.ch1_slider.setToolTip(
            "Adjust the weighting of the first channel."
        )
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
        # Show tick marks on the channel slider
        self.ch2_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ch2_slider.setTickInterval(10)
        self.ch2_slider.setToolTip(
            "Adjust the weighting of the second channel."
        )
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
        # Show tick marks on the channel slider
        self.ch3_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ch3_slider.setTickInterval(10)
        self.ch3_slider.setToolTip(
            "Adjust the weighting of the third channel."
        )
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
        self.min_threshold_spinbox.setToolTip(
            "Minimum intensity value for the mask. Pixels with values below this are excluded from the binary mask."
        )

        max_thresh_label = QLabel("Max Threshold:")
        self.max_threshold_spinbox = QDoubleSpinBox()
        self.max_threshold_spinbox.setRange(0.0, 1.0)
        self.max_threshold_spinbox.setSingleStep(0.01)
        self.max_threshold_spinbox.setValue(1.0)
        self.max_threshold_spinbox.setToolTip(
            "Maximum intensity value for the mask. Pixels with values above this are excluded from the binary mask."
        )

        threshold_layout.addWidget(min_thresh_label)
        threshold_layout.addWidget(self.min_threshold_spinbox)
        threshold_layout.addWidget(max_thresh_label)
        threshold_layout.addWidget(self.max_threshold_spinbox)
        sliders_layout.addLayout(threshold_layout)

        # Load and process buttons
        buttons_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Image")
        buttons_layout.addWidget(self.load_button)
        sliders_layout.addLayout(buttons_layout)

        # Add sliders to controls
        controls_layout.addLayout(sliders_layout)

        # Ensure the controls panel stays compact
        controls_container = QWidget()
        controls_container.setLayout(controls_layout)
        controls_container.setMaximumHeight(180)  # limit height
        controls_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # fixed vertical size

        # Add controls to main layout
        main_layout.addWidget(controls_container)

        # --------------------------------------------------------------
        # Image display area with a navigation toolbar above the canvas
        # --------------------------------------------------------------
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # canvas expands
        main_layout.addWidget(self.canvas)

        # Toolbar gives Zoom In, Zoom Out, Pan, Home, Save, etc.
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Mouse‑wheel zoom (adds a smoother, intuitive zoom)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        # Insert toolbar then canvas into the vertical layout
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

        # Make the layout responsive: give canvas the remaining stretch
        main_layout.setStretch(0, 0)  # controls with controls (index 0) – no stretch
        main_layout.setStretch(1, 0)  # toolbar
        main_layout.setStretch(2, 1)  # canvas (index 2) – occupies extra space

        # Connect signals
        self.ch1_slider.valueChanged.connect(self.update_ch1_value)
        self.ch1_slider.valueChanged.connect(self.process_image)
        self.ch2_slider.valueChanged.connect(self.update_ch2_value)
        self.ch2_slider.valueChanged.connect(self.process_image)
        self.ch3_slider.valueChanged.connect(self.update_ch3_value)
        self.ch3_slider.valueChanged.connect(self.process_image)
        self.color_space_combo.currentTextChanged.connect(self.process_image)
        self.min_threshold_spinbox.valueChanged.connect(self.process_image)
        self.max_threshold_spinbox.valueChanged.connect(self.process_image)
        self.load_button.clicked.connect(self.load_image)

    def _on_scroll(self, event):
        """Zoom all three axes with the mouse wheel."""
        if not self.figure.axes:
            return

        # Find which axes the event occurred in
        target_ax = None
        for ax in self.figure.axes:
            if ax.contains(event)[0]:
                target_ax = ax
                break

        if target_ax is None or event.xdata is None or event.ydata is None:
            return  # ignore scrolls outside the image area

        # Determine zoom direction
        scale_factor = 1.1 if event.button == 'up' else 0.9

        # Current limits of the target axes
        cur_xlim = target_ax.get_xlim()
        cur_ylim = target_ax.get_ylim()

        # Compute new limits keeping the mouse position stationary
        xdata, ydata = event.xdata, event.ydata
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

        new_xlim = [xdata - relx * new_width, xdata + (1 - relx) * new_width]
        new_ylim = [ydata - rely * new_height, ydata + (1 - rely) * new_height]

        # Apply the same limits to all axes
        for ax in self.figure.axes:
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)

        self.canvas.draw_idle()

    def show_color_space_help(self):
        """Display a dialog explaining the available color spaces and channel effects."""
        help_text = (
            "<b>Color Spaces</b><br><br>"
            "<li><b>RGB</b>: Red, Green, Blue channels. "
            "Adjusting a channel changes the contribution of that color component to the linear filter.</li>"
            "<li><b>HSV</b>: Hue, Saturation, Value. "
            "Hue controls the dominant color, Saturation controls color intensity, and Value controls brightness. "
            "Changing a slider modifies the weight of the selected component in the filtered grayscale image.</li>"
            "<li><b>YCbCr</b>: Luminance (Y), Blue‑difference (Cb), Red‑difference (Cr). "
            "Y represents brightness, while Cb and Cr carry color difference information. "
            "Adjusting these sliders influences how bright or colored features are emphasized in the filtered result.</li>"
            "<br><b>Effect on filtered image</b><br>"
            "The three sliders provide coefficients (0-1) for the chosen channels. "
            "The filtered image is computed as a linear combination of the selected channels using the provided weights. "
            "Increasing a channel’s coefficient makes features prominent in that channel appear brighter in the grayscale "
            "<i>image</i>, which in turn affects the binary mask generated by the threshold controls."
        )
        QMessageBox.information(self, "Color Space Information", help_text)

    def update_channel_labels(self, mode):
        """Update channel‐label texts according to the selected color space.

        Parameters
        ----------
        mode : str
            The color space selected in the combo box. Accepted values are ``'RGB'``, ``'HSV'`` and ``'YCbCr'``.
        """
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
        """Refresh the displayed value for channel 1.

        Parameters
        ----------
        value : int
            Slider position in the range ``0``–``100``.
            The displayed coefficient is ``value / 100``.
        """
        self.ch1_value.setText(f"{value / 100:.2f}")

    def update_ch2_value(self, value):
        """Refresh the displayed value for channel 2.

        Parameters
        ----------
        value : int
            Slider position in the range ``0``–``100``.
        """
        self.ch2_value.setText(f"{value / 100:.2f}")

    def update_ch3_value(self, value):
        """Refresh the displayed value for channel 3.

        Parameters
        ----------
        value : int
            Slider position in the range ``0``–``100``.

        """
        self.ch3_value.setText(f"{value / 100:.2f}")

    def load_image(self):
        """Open a file‑dialog, load an image, and display it.

        The image is converted to ``RGB`` and normalized to ``[0, 1]``.
        If loading succeeds, the *Process* button becomes enabled.

        Raises
        ------
        Exception
            Any exception raised by `PIL.Image.Image.open` is caught and printed to stdout.
        """
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

                # Process image immediately after loading
                self.process_image()
            except Exception as e:
                print(f"Error loading image: {e}")

    def load_image_from_path(self, file_path: str) -> None:
        """Load an image from an absolute path (used for CLI start‑up).

        Parameters
        ----------
        file_path : str
            Absolute path to an image file supported by Pillow.

        Raises
        ------
        Exception
            Propagates any error raised while opening or converting the file.
        """
        if file_path:
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

                # Process image immediately after loading
                self.process_image()
            except Exception as e:
                print(f"Error loading image from path '{file_path}': {e}")

    def process_image(self):
        """Apply the linear filter and threshold, then display results.

        The method reads the current slider positions to obtain channel coefficients, converts the source image
        to the selected color space (if needed), calls `plant3dvision.proc2d.linear` and finally creates a binary
        mask based on the user‑defined thresholds.

        Notes
        -----
        The function updates three sub‑plots:
        * original image,
        * filtered grayscale image,
        * binary mask.
        """
        if self.original_img is None:
            return

        # Save current axis limits before clearing (if axes exist)
        saved_xlim, saved_ylim = None, None
        if self.figure.axes:
            saved_xlim = self.figure.axes[0].get_xlim()
            saved_ylim = self.figure.axes[0].get_ylim()

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

        # Restore saved axis limits (pan/zoom state)
        if saved_xlim is not None and saved_ylim is not None:
            for ax in self.figure.axes:
                ax.set_xlim(saved_xlim)
                ax.set_ylim(saved_ylim)

        self.canvas.draw()

        # Synchronize axes limits for pan/zoom (only sets up callbacks)
        self._sync_axes_limits()

    def _sync_axes_limits(self):
        """Synchronize the pan and zoom limits across all three axes."""
        if len(self.figure.axes) < 3:
            return

        # Use the first axes as the reference
        ref_ax = self.figure.axes[0]
        xlim = ref_ax.get_xlim()
        ylim = ref_ax.get_ylim()

        # Apply to all other axes
        for ax in self.figure.axes[1:]:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        # Connect pan/zoom events to sync handler
        for ax in self.figure.axes:
            ax.callbacks.connect('xlim_changed', self._on_axes_changed)
            ax.callbacks.connect('ylim_changed', self._on_axes_changed)

    def _on_axes_changed(self, ax):
        """Callback when any axes limits change (pan/zoom via toolbar)."""
        if not hasattr(self, '_syncing') or not self._syncing:
            self._syncing = True
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Apply to all axes
            for other_ax in self.figure.axes:
                if other_ax != ax:
                    other_ax.set_xlim(xlim)
                    other_ax.set_ylim(ylim)

            self.canvas.draw_idle()
            self._syncing = False


@click.command()
@click.argument('image_path', required=False, type=click.Path(exists=True, dir_okay=False))
def main(image_path: str | None = None):
    """Start the RGB linear filter GUI.

    Optionally, provide an image file path to load at launch.
    """
    app = QApplication(sys.argv)
    window = RGBFilterApp()
    if image_path:
        window.load_image_from_path(image_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
