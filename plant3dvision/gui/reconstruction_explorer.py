import os

if os.getenv("QT_QPA_PLATFORM") is None:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import sys
import argparse

import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QCheckBox
from PySide6.QtWidgets import QColorDialog
from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QSlider
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from plantdb.commons.fsdb.core import FSDB
from plantdb.commons.io import read_point_cloud
from plantdb.server.core.utils import compute_fileset_matches
from romitask.log import get_logger

logger = get_logger(__file__.split('/')[-1].split('.')[0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _camera_params_from_file(image_f):
    """Return (cam_pos, focal_point, up, fov_y_deg) from a ROMI image file."""
    img_md = image_f.get_metadata('colmap_camera')
    x, y, z, _, _, _ = image_f.get_metadata('estimated_pose')
    rot_mat = np.array(img_md['rotmat'])
    fy = img_md['camera_model']['params'][1]
    img_height = img_md.get('height') or Image.open(image_f.path()).size[1]

    rot_mat_inv = rot_mat.T
    cam_pos = (x, y, z)
    forward_world = rot_mat_inv[:, 2]
    up_world = -rot_mat_inv[:, 1]
    focal_point = np.array(cam_pos) + forward_world
    fov_y_deg = np.degrees(2.0 * np.arctan(img_height / (2.0 * fy)))
    return cam_pos, tuple(focal_point), tuple(up_world), fov_y_deg


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ReconstructionExplorer(QMainWindow):

    def __init__(self, db_path: str):
        super().__init__()
        self.setWindowTitle("Reconstruction Explorer")
        self.resize(1400, 800)

        # DB connection
        self._db = FSDB(db_path)
        self._db.connect()
        self._db.login('guest', 'guest')

        # State
        self._cloud_actor = None
        self._images_fs = None
        self._image_files = []
        self._point_color = "dodgerblue"
        self._grid_visible = True
        self._current_cam_params = None  # (pos, focal, up, fov)

        self._build_ui()
        self._populate_scan_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # -- Left panel (80%) ------------------------------------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # PyVista interactor
        self.plotter = pvqt.QtInteractor(left)
        # set a dark‑grey background (RGB values range from 0‑1)
        self.plotter.set_background([0.15, 0.15, 0.15])
        self.plotter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plotter.add_text(
            "Select a scan ID from the dropdown",
            position="upper_left",
            font_size=12,
            color="white",
            name="hint_text",
        )
        left_layout.addWidget(self.plotter.interactor, stretch=1)

        # Slider row
        slider_row = QHBoxLayout()
        self._slider_label = QLabel("Image: -")
        self._slider_label.setFixedWidth(160)
        self._image_slider = QSlider(Qt.Orientation.Horizontal)
        self._image_slider.setEnabled(False)
        self._image_slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self._slider_label)
        slider_row.addWidget(self._image_slider)
        left_layout.addLayout(slider_row)

        root.addWidget(left, stretch=8)

        # -- Right panel (20%) -----------------------------------------
        right = QWidget()
        right.setFixedWidth(260)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(8)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Scan selector
        scan_group = QGroupBox("Scan")
        scan_vbox = QVBoxLayout(scan_group)
        self._scan_combo = QComboBox()
        self._scan_combo.maxVisibleItems = 10
        self._scan_combo.currentIndexChanged.connect(self._on_scan_selected)
        scan_vbox.addWidget(self._scan_combo)
        right_layout.addWidget(scan_group)

        # Point-cloud color
        color_group = QGroupBox("Point Cloud Color")
        color_vbox = QVBoxLayout(color_group)
        self._color_btn = QPushButton("Pick color")
        self._color_btn.setEnabled(False)
        self._color_btn.clicked.connect(self._on_pick_color)
        color_vbox.addWidget(self._color_btn)
        right_layout.addWidget(color_group)

        # Grid toggle
        grid_group = QGroupBox("Display")
        grid_vbox = QVBoxLayout(grid_group)
        self._grid_checkbox = QCheckBox("Show grid")
        self._grid_checkbox.setChecked(True)
        self._grid_checkbox.stateChanged.connect(self._on_grid_toggled)
        grid_vbox.addWidget(self._grid_checkbox)
        right_layout.addWidget(grid_group)

        # Camera / snapshot
        cam_group = QGroupBox("Camera")
        cam_vbox = QVBoxLayout(cam_group)
        self._reset_cam_btn = QPushButton("Reset to image camera")
        self._reset_cam_btn.setEnabled(False)
        self._reset_cam_btn.clicked.connect(self._on_reset_camera)
        self._snapshot_btn = QPushButton("Take snapshot")
        self._snapshot_btn.setEnabled(False)
        self._snapshot_btn.clicked.connect(self._on_snapshot)
        cam_vbox.addWidget(self._reset_cam_btn)
        cam_vbox.addWidget(self._snapshot_btn)
        right_layout.addWidget(cam_group)

        right_layout.addStretch()
        root.addWidget(right, stretch=2)

    def _populate_scan_list(self):
        self._scan_combo.blockSignals(True)
        self._scan_combo.addItem("- select a scan -", userData=None)
        scan_list = self._db.list_scans(owner_only=False)
        logger.info(f"Loaded scan list: {scan_list}")
        for scan_id in sorted(scan_list):
            active = True
            fs_matches = compute_fileset_matches(self._db.get_scan(scan_id))
            if "PointCloud" not in fs_matches:
                active = False
            self._scan_combo.addItem(scan_id, userData=scan_id)
            # Disable the entry when it isn’t active
            if not active:
                row = self._scan_combo.count() - 1  # the row we just added
                model = self._scan_combo.model()
                item = model.item(row)  # QStandardItem
                item.setEnabled(False)  # greys‑out / makes it unselectable
        self._scan_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_scan_selected(self, index: int):
        scan_id = self._scan_combo.itemData(index)
        if scan_id is None:
            return
        self._load_scan(scan_id)

    def _on_slider_changed(self, value: int):
        if not self._image_files:
            return
        image_f = self._image_files[value]
        self._slider_label.setText(f"Image: {image_f.id}")
        self._apply_image(image_f)

    def _on_pick_color(self):
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            hex_color = color.name()  # e.g. "#ff8800"
            self._point_color = hex_color
            self._apply_point_color(hex_color)

    def _on_reset_color(self):
        self._point_color = "dodgerblue"
        self._apply_point_color("dodgerblue")

    def _on_grid_toggled(self, state):
        if state == Qt.CheckState.Checked.value:
            self.plotter.show_grid(color='white')
        else:
            self.plotter.remove_bounds_axes()
        self.plotter.render()

    def _on_reset_camera(self):
        if self._current_cam_params is None:
            return
        pos, focal, up, fov = self._current_cam_params
        cam = self.plotter.camera
        cam.position = pos
        cam.focal_point = focal
        cam.up = up
        cam.view_angle = fov
        cam.disable_parallel_projection()
        self.plotter.render()

    def _on_snapshot(self):
        scan_id = self._scan_combo.currentData()
        filename = f"snapshot_{scan_id or 'scene'}.png"
        self.plotter.screenshot(filename)
        self.statusBar().showMessage(f"Snapshot saved → {filename}", 4000)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_scan(self, scan_id: str):
        scan = self._db.get_scan(scan_id)
        fs_matches = compute_fileset_matches(scan)

        # Point cloud
        pcd_fs = scan.get_fileset(fs_matches['PointCloud'])
        pcd_f = pcd_fs.get_file('PointCloud')
        pcd = read_point_cloud(pcd_f.path())
        pcd_arr = np.asarray(pcd.points)

        self._cloud = pv.PolyData(pcd_arr)

        # Images fileset
        self._images_fs = scan.get_fileset('images')
        self._image_files = self._images_fs.get_files()

        # Reset plotter
        self.plotter.clear()
        self._cloud_actor = None

        self._render_point_cloud(self._point_color)

        if self._grid_visible:
            self.plotter.show_grid(color='white')

        # Setup slider
        n = len(self._image_files)
        self._image_slider.blockSignals(True)
        self._image_slider.setMinimum(0)
        self._image_slider.setMaximum(max(0, n - 1))
        self._image_slider.setValue(0)
        self._image_slider.setEnabled(n > 0)
        self._image_slider.blockSignals(False)

        if n > 0:
            self._apply_image(self._image_files[0])

        # Enable controls
        self._color_btn.setEnabled(True)
        self._reset_cam_btn.setEnabled(True)
        self._snapshot_btn.setEnabled(True)

    def _render_point_cloud(self, color):
        """Add (or replace) the point cloud actor."""
        if self._cloud_actor is not None:
            self.plotter.remove_actor(self._cloud_actor)
            self._cloud_actor = None

        common = dict(point_size=2, render_points_as_spheres=False, style="points")

        c = color
        self._cloud_actor = self.plotter.add_mesh(self._cloud, color=c, **common)

        self.plotter.render()

    def _apply_point_color(self, color):
        self._render_point_cloud(color)

    def _apply_image(self, image_f):
        """Update background image and camera to match the selected image."""
        try:
            cam_pos, focal_point, up_world, fov_y_deg = _camera_params_from_file(image_f)
        except Exception:
            # Metadata missing – just show the image, skip camera update
            cam_pos = focal_point = up_world = fov_y_deg = None

        try:
            self.plotter.add_background_image(image_f.path(), as_global=False)
        except:
            self.plotter.remove_background_image()
            self.plotter.add_background_image(image_f.path(), as_global=False)

        self.plotter.add_text(
            f"Camera: {image_f.id}",
            position="upper_left",
            font_size=10,
            color="white",
            name="cam_label",
        )

        if cam_pos is not None:
            self._current_cam_params = (cam_pos, focal_point, up_world, fov_y_deg)
            cam = self.plotter.camera
            cam.position = cam_pos
            cam.focal_point = focal_point
            cam.up = up_world
            cam.view_angle = fov_y_deg
            cam.disable_parallel_projection()

        # Ensure the renderer doesn't clip the point cloud
        self.plotter.renderer.ResetCameraClippingRange()

        self.plotter.render()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="ROMI Reconstruction Explorer")
    parser.add_argument("--db", default=None, help="Path to ROMI FSDB database")
    return parser.parse_args()


def main():
    args = _parse_args()
    db_path = args.db or os.environ.get("ROMI_DB")
    if not db_path:
        logger.error("Provide --db <path> or set the ROMI_DB environment variable.")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = ReconstructionExplorer(db_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
