#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**RGB Linear Filter GUI**

A Python module that launches an interactive Qt‑based application for loading plant scan images, applying a customizable linear combination of color‑space channels, and visualizing the filtered result together with a threshold‑derived binary mask.
It streamlines the exploration of channel weighting and threshold parameters, making it easy to fine‑tune image preprocessing for downstream analysis.

## Key Features

- Load images from a PlantDB (FSDB) scan.
- Select among three color spaces (RGB, HSV, YCbCr) and adjust each channel’s contribution with sliders.
- Real‑time preview of the original image, the filtered grayscale image, and the binary mask.
- Interactive threshold controls (min / max) and optional binary dilation.
- Synchronized pan/zoom across all three sub‑plots, with mouse‑wheel zoom support.
- Export the current filter parameters to a `local_config.toml` file attached to the chosen scan.
- Search and filter scans via a searchable dropdown.

## Usage Examples

```shell
# Launch the GUI, automatically discovering the FSDB path from the 'ROMI_DB' environment variable
linear_filter

# Or provide an explicit FSDB directory and optionally open a specific scan
linear_filter /path/to/FSDB --scan SCAN_ID
```

Running the script opens the window where you can browse scans, adjust color‑space sliders, set threshold limits, and instantly see how the linear filter affects the image and its mask.
When satisfied, click **Export Parameters** to save the configuration back to the scan’s `local_config.toml`.
"""

import os
import sys
from pathlib import Path

import click
import numpy as np
import tomlkit
from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtCore import Qt
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QDoubleSpinBox
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QSlider
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from plantdb.commons.fsdb.core import FSDB
from plantdb.commons.log import DEFAULT_LOG_LEVEL
from plantdb.commons.log import LOG_LEVELS
from plantdb.commons.log import get_logger

from plant3dvision.proc2d import dilation
from plant3dvision.proc2d import linear

# Create a logger and set the environment variable
logger = get_logger("LinearFilterApp", log_level=DEFAULT_LOG_LEVEL)


class LinearFilterApp(QMainWindow):
    """Linear filter GUI application.

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

    def __init__(self, fsdb_path: str | Path, scan_id: str | None):
        """Initialize the main window and UI elements.

        Set the window title, geometry and creates placeholders for image data and channel coefficients.
        Then call `initUI` to build the graphical user interface.
        """
        super().__init__()
        self.setWindowTitle("Linear Filter and Threshold")
        self.setGeometry(100, 100, 1200, 800)

        # Database connection
        self.fsdb_path = fsdb_path
        self.db = None
        self.scan_ids_list = []
        self.images_list = []
        self.current_scan = None

        # Image placeholders
        self.source_image: Image = None  # PIL source image
        self.original_img = None
        self.filtered_img = None
        self.mask = None

        self.ch1_value = None
        self.ch2_value = None
        self.ch3_value = None

        # Initialize database
        self._init_database()

        # Timers (wait after modifications stop for 200 ms before processing)
        self._load_image_timer = QTimer(self, singleShot=True, interval=300)
        self._load_image_timer.timeout.connect(self._load_image)
        self._process_image_timer = QTimer(self, singleShot=True, interval=300)
        self._process_image_timer.timeout.connect(self.process_image)

        # Initialize UI
        self.initUI()

        if scan_id and scan_id in self.scan_ids_list:
            self.scan_dropdown.setCurrentIndex(self.scan_ids_list.index(scan_id))
            self._load_scan(scan_id)

    def _init_database(self):
        """Initialize the database connection and load scan list."""
        try:
            self.db = FSDB(self.fsdb_path, no_auth=True)
            self.db.connect()
            self.scan_ids_list = self.db.list_scans(owner_only=False)
            logger.info(f"Loaded {len(self.scan_ids_list)} scans from the FSDB")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            self.scan_ids_list = []

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

        # ===== TOP PANEL =====
        top_panel_layout = QHBoxLayout()

        # Search box
        search_label = QLabel("Search:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter scans...")
        self.search_box.textChanged.connect(self._filter_scans)
        self.search_box.setMinimumWidth(150)
        self.search_box.setMaximumWidth(200)
        self.search_box.setToolTip(
            "Type to filter the list of scans. Only scans containing the entered text will be shown."
        )
        top_panel_layout.addWidget(search_label)
        top_panel_layout.addWidget(self.search_box)

        # Scan dropdown
        scan_label = QLabel("Scan:")
        self.scan_dropdown = QComboBox()
        self.scan_dropdown.addItems(self.scan_ids_list)
        self.scan_dropdown.currentTextChanged.connect(self._load_scan)
        top_panel_layout.addWidget(scan_label)
        top_panel_layout.addWidget(self.scan_dropdown)

        # Image slider
        image_slider_label = QLabel("Image:")
        self.image_slider = QSlider(Qt.Orientation.Horizontal)
        self.image_slider.setMinimum(0)
        self.image_slider.setMaximum(0)
        self.image_slider.setValue(0)
        # Show tick marks on the image index slider
        self.image_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.image_slider.setTickInterval(1)
        self.image_slider.valueChanged.connect(self._load_image_from_slider)
        # Replace label with a QDoubleSpinBox for precise image index control
        self.image_index_spinbox = QDoubleSpinBox()
        self.image_index_spinbox.setDecimals(0)
        self.image_index_spinbox.setSingleStep(1.0)
        self.image_index_spinbox.setMinimum(0)
        self.image_index_spinbox.setMaximum(0)
        self.image_index_spinbox.setValue(0)
        # When the spinbox value changes, update the slider accordingly
        self.image_index_spinbox.valueChanged.connect(self._on_image_index_spinbox_changed)
        top_panel_layout.addWidget(image_slider_label)
        top_panel_layout.addWidget(self.image_slider)
        top_panel_layout.addWidget(self.image_index_spinbox)

        top_panel_container = QWidget()
        top_panel_container.setLayout(top_panel_layout)
        top_panel_container.setMaximumHeight(50)
        main_layout.addWidget(top_panel_container)

        # ===== CONTROLS PANEL =====
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

        # Threshold & Dilation controls
        threshold_dilation_layout = QHBoxLayout()

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

        # Dilation control
        dilation_label = QLabel("Dilation:")
        self.dilation_spinbox = QDoubleSpinBox()
        self.dilation_spinbox.setRange(0, 5)
        self.dilation_spinbox.setValue(0)
        # Show a helpful tooltip when the user hovers over the export button
        self.dilation_spinbox.setToolTip(
            "Binary dilation applied to the mask image."
        )

        threshold_dilation_layout.addWidget(min_thresh_label)
        threshold_dilation_layout.addWidget(self.min_threshold_spinbox)
        threshold_dilation_layout.addWidget(max_thresh_label)
        threshold_dilation_layout.addWidget(self.max_threshold_spinbox)
        threshold_dilation_layout.addWidget(dilation_label)
        threshold_dilation_layout.addWidget(self.dilation_spinbox)
        sliders_layout.addLayout(threshold_dilation_layout)

        # Export Parameters button
        self.export_button = QPushButton("Export Parameters")
        self.export_button.setMinimumWidth(250)
        self.export_button.clicked.connect(self._export_parameters)
        # Show a helpful tooltip when the user hovers over the export button
        self.export_button.setToolTip(
            "Export the current parameters to a local configuration for the selected scan."
        )
        sliders_layout.addWidget(self.export_button, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Add sliders to controls
        controls_layout.addLayout(sliders_layout)

        # Ensure the controls panel stays compact
        controls_container = QWidget()
        controls_container.setLayout(controls_layout)
        controls_container.setMaximumHeight(250)  # limit height
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
        main_layout.setStretch(0, 0)  # top panel (index 0) – no stretch
        main_layout.setStretch(1, 0)  # controls (index 1) – no stretch
        main_layout.setStretch(2, 0)  # toolbar
        main_layout.setStretch(3, 1)  # canvas (index 3) – occupies extra space

        # Connect signals
        self.ch1_slider.valueChanged.connect(self.update_ch1_value)
        self.ch1_slider.valueChanged.connect(self._process_image_timer.start)
        self.ch2_slider.valueChanged.connect(self.update_ch2_value)
        self.ch2_slider.valueChanged.connect(self._process_image_timer.start)
        self.ch3_slider.valueChanged.connect(self.update_ch3_value)
        self.ch3_slider.valueChanged.connect(self._process_image_timer.start)
        self.color_space_combo.currentTextChanged.connect(self._process_image_timer.start)
        self.min_threshold_spinbox.valueChanged.connect(self._process_image_timer.start)
        self.max_threshold_spinbox.valueChanged.connect(self._process_image_timer.start)
        self.dilation_spinbox.valueChanged.connect(self._process_image_timer.start)

    def _filter_scans(self, text):
        """Filter the scan dropdown based on search box text."""
        self.scan_dropdown.clear()
        filtered_scans = [scan_id for scan_id in self.scan_ids_list if text.lower() in scan_id.lower()]
        self.scan_dropdown.addItems(filtered_scans)

    def _load_scan(self, scan_id):
        """Load the selected scan and populate the image slider."""
        if not scan_id or not self.db:
            return

        try:
            self.current_scan = self.db.get_scan(scan_id)
            images_fs = self.current_scan.get_fileset('images')
            self.images_list = images_fs.get_files()

            # Update image slider
            self.image_slider.setMaximum(max(0, len(self.images_list) - 1))
            self.image_slider.setValue(0)
            self._update_image_label()

            # Load first image
            if self.images_list:
                self._load_image_from_slider(0)
            self._import_parameters()

        except Exception as e:
            logger.error(f"Error loading scan '{scan_id}': {e}")
            self.images_list = []
            self.image_slider.setMaximum(0)
            self._update_image_label()

    @Slot()
    def _load_image_from_slider(self, index):
        """Select the image at the given slider index and start load timer."""
        if not self.images_list or index >= len(self.images_list):
            return

        try:
            image = self.images_list[index]
            self._image_path = image.path()
            self._load_image_timer.start()
        except Exception as e:
            logger.error(f"Error loading image at index {index}: {e}")

    @Slot()
    def _load_image(self):
        """Load the image."""
        try:
            self.load_image_from_path(self._image_path)
            self._update_image_label()
        except Exception as e:
            logger.error(f"Error loading image at index {int(self.image_index_spinbox.value())}: {e}")

    def _update_image_label(self):
        """Update the image index label."""
        if self.images_list:
            # Use 1‑based indexing for the spinbox display
            current = self.image_slider.value() + 1
            total = len(self.images_list)
            # Update spinbox range and suffix (e.g., "5/20")
            self.image_index_spinbox.blockSignals(True)
            self.image_index_spinbox.setMinimum(1)
            self.image_index_spinbox.setMaximum(total)
            self.image_index_spinbox.setValue(current)
            self.image_index_spinbox.setSuffix(f"/{total}")
            self.image_index_spinbox.blockSignals(False)
        else:
            self.image_index_spinbox.blockSignals(True)
            self.image_index_spinbox.setMinimum(0)
            self.image_index_spinbox.setMaximum(0)
            self.image_index_spinbox.setValue(0)
            self.image_index_spinbox.setSuffix("")
            self.image_index_spinbox.blockSignals(False)

    def _on_image_index_spinbox_changed(self, value):
        """Synchronize slider when the image index spinbox changes."""
        if not self.images_list:
            return
        # Convert 1‑based spinbox value to 0‑based slider index

        new_index = int(value) - 1
        # Clamp to valid range just in case
        new_index = max(0, min(new_index, len(self.images_list) - 1))
        self.image_slider.setValue(new_index)

    def _import_parameters(self):
        """import current parameters from local_config.toml file, if any."""
        config_path = self.current_scan.path() / "local_config.toml"
        try:
            with open(config_path, "rb") as f:
                existing_config = tomlkit.load(f)
        except Exception as e:
            existing_config = {}

        mask_cfg = existing_config.get("Masks")
        if not mask_cfg:
            return  # nothing to load

        logger.info("Found a local_config.toml file, loading previous parameters...")
        # - Make sure we have a list of three float coefficients
        params = mask_cfg.get("parameters", [])
        if isinstance(params, str):
            # Stored as a string like "[0.5, 1.0, 0.5]"
            params = [float(v) for v in params.strip("[]").split(",")]
        elif isinstance(params, list):
            # Ensure every element is a float (it may be an int)
            params = [float(v) for v in params]

        # - Populate the UI widgets
        # colour‑space selector
        self.color_space_combo.setCurrentText(mask_cfg.get("colorspace", "RGB"))

        # sliders expect values in the range 0–100
        self.ch1_slider.setValue(int(params[0] * 100))
        self.ch2_slider.setValue(int(params[1] * 100))
        self.ch3_slider.setValue(int(params[2] * 100))

        # thresholds and dilation
        self.min_threshold_spinbox.setValue(mask_cfg.get("min_threshold", 0.0))
        self.max_threshold_spinbox.setValue(mask_cfg.get("max_threshold", 1.0))
        self.dilation_spinbox.setValue(mask_cfg.get("dilation", 0))

    def _export_parameters(self):
        """Export current parameters to local_config.toml file."""
        config_path = self.current_scan.path() / "local_config.toml"
        try:
            with open(config_path, "rb") as f:
                existing_config = tomlkit.load(f)
        except Exception as e:
            existing_config = {}

        try:
            # Update with new mask parameters
            existing_config["Masks"] = {
                "method": "linear",
                "colorspace": self.color_space_combo.currentText(),
                "parameters": [
                    self.ch1_slider.value() / 100.0,
                    self.ch2_slider.value() / 100.0,
                    self.ch3_slider.value() / 100.0
                ],
                "min_threshold": self.min_threshold_spinbox.value(),
                "max_threshold": self.max_threshold_spinbox.value(),
                "dilation": self.dilation_spinbox.value(),
            }
            # Convert the list of coefficients to a string
            existing_config["Masks"]["parameters"] = str(existing_config["Masks"]["parameters"])
            # Write to file
            with open(config_path, "w") as f:
                tomlkit.dump(existing_config, f)

            logger.info(f"Parameters exported to {config_path}")

        except Exception as e:
            logger.error(f"Error exporting parameters: {e}")

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
                # Process image immediately after loading
                self.process_image()
            except Exception as e:
                logger.error(f"Error loading image from path '{file_path}': {e}")

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
        dilation_iterations = self.dilation_spinbox.value()
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

        # Apply dilation if needed
        if dilation_iterations > 0:
            self.mask = dilation(self.mask, int(dilation_iterations))

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
        title = f"Mask ({min_threshold:.2f} <= v <= {max_threshold:.2f})"
        if dilation_iterations > 0:
            title += f"\nDilation: {dilation_iterations}"
        ax3.set_title(title)
        ax3.axis('off')

        self.figure.suptitle(f"{self.current_scan.id} - {self.images_list[self.image_slider.value()].id}")
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
@click.argument('fsdb_path', required=False, type=click.Path(exists=True, dir_okay=True))
@click.option('-s', '--scan', 'scan_id', type=str)
def main(fsdb_path: str | None = None, scan_id: str = None):
    """Start the RGB linear filter GUI.

    Optionally, provide an image file path to load at launch.
    """
    # Try to use the 'ROMI_DB' environment variable as path to the FSDB if not set as argument.
    if not fsdb_path:
        fsdb_path = os.environ.get('ROMI_DB', None)
    if not fsdb_path:
        raise ValueError(f"Provide a valid path to an FSDB folder or set 'ROMI_DB' environment variable.")

    app = QApplication(sys.argv)
    window = LinearFilterApp(fsdb_path, scan_id)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
