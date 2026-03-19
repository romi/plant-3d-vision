import os

import toml
from PySide6.QtWidgets import QSpinBox
from matplotlib import pyplot as plt

from plant3dvision.visu import pyvista_mesh
from plant3dvision.visu import pyvista_volume
from plantdb.commons.fsdb.exceptions import FilesetNotFoundError
from plantdb.commons.io import read_triangle_mesh
from plantdb.commons.io import read_volume

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
        self._pcd_actor = None
        self._vol_actor = None
        self._mesh_actor = None
        self._vol = None
        self._pcd = None
        self._mesh = None
        self._origin = None
        self._spacing = None
        self._vol_path = None
        self._pcd_path = None
        self._mesh_path = None
        self._images_fs = None
        self._image_files = []
        self._voxel_colormap = "inferno"
        self._pcd_color = "dodgerblue"
        self._pcd_opacity = 1.0  # fully opaque
        self._pcd_point_size = 2  # default point size
        self._mesh_color = "neon"
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
        # Show tick marks below the slider (you can also use TicksAbove / TicksBothSides)
        self._image_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        # One tick per step to match the integer range of the slider
        self._image_slider.setTickInterval(1)
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

        # Voxels data
        voxel_group = QGroupBox("Voxels")
        voxel_vbox = QVBoxLayout(voxel_group)
        self._voxel_checkbox = QCheckBox("Show voxels")
        self._voxel_checkbox.setEnabled(False)
        self._voxel_checkbox.stateChanged.connect(self._on_voxel_toggled)
        # Dropdown that lists all PyVista colormaps
        self._voxel_cmap_combo = QComboBox()
        self._voxel_cmap_combo.setEnabled(False)
        # Populate with the names of the available colormaps
        self._voxel_cmap_combo.addItems(plt.colormaps)
        self._voxel_cmap_combo.setCurrentText(self._voxel_colormap)
        self._voxel_cmap_combo.currentTextChanged.connect(self._on_voxel_cmap_changed)
        voxel_vbox.addWidget(self._voxel_checkbox)
        voxel_vbox.addWidget(self._voxel_cmap_combo)
        right_layout.addWidget(voxel_group)

        # Point-cloud data
        pcd_group = QGroupBox("Point Cloud")
        pcd_vbox = QVBoxLayout(pcd_group)
        self._pcd_checkbox = QCheckBox("Show point cloud")
        self._pcd_checkbox.setEnabled(False)
        self._pcd_checkbox.stateChanged.connect(self._on_pcd_toggled)
        self._pcd_color_btn = QPushButton("Pick color")
        self._pcd_color_btn.setEnabled(False)
        self._pcd_color_btn.clicked.connect(self._on_pick_pcd_color)
        # Opacity slider (0‑100 → 0.0‑1.0)
        self._pcd_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._pcd_opacity_slider.setRange(0, 100)
        self._pcd_opacity_slider.setValue(100)  # fully opaque by default
        self._pcd_opacity_slider.setEnabled(False)
        self._pcd_opacity_slider.valueChanged.connect(self._on_pcd_opacity_changed)
        # Point‑size spin box (1‑50)
        self._pcd_point_spin = QSpinBox()
        self._pcd_point_spin.setRange(1, 50)
        self._pcd_point_spin.setValue(2)  # matches the default in _render_point_cloud
        self._pcd_point_spin.setEnabled(False)
        self._pcd_point_spin.valueChanged.connect(self._on_pcd_point_size_changed)
        pcd_vbox.addWidget(self._pcd_checkbox)
        pcd_vbox.addWidget(self._pcd_color_btn)
        pcd_vbox.addWidget(QLabel("Opacity"))
        pcd_vbox.addWidget(self._pcd_opacity_slider)
        pcd_vbox.addWidget(QLabel("Point size"))
        pcd_vbox.addWidget(self._pcd_point_spin)
        right_layout.addWidget(pcd_group)

        # Triangular Mesh data
        mesh_group = QGroupBox("Triangular Mesh")
        mesh_vbox = QVBoxLayout(mesh_group)
        self._mesh_checkbox = QCheckBox("Show mesh")
        self._mesh_checkbox.setEnabled(False)
        self._mesh_checkbox.stateChanged.connect(self._on_mesh_toggled)
        self._mesh_color_btn = QPushButton("Pick color")
        self._mesh_color_btn.setEnabled(False)
        self._mesh_color_btn.clicked.connect(self._on_pick_mesh_color)
        mesh_vbox.addWidget(self._mesh_checkbox)
        mesh_vbox.addWidget(self._mesh_color_btn)
        right_layout.addWidget(mesh_group)

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
    # Toggle slots
    # ------------------------------------------------------------------

    def _on_voxel_toggled(self, state):
        if state == Qt.CheckState.Checked.value:
            if self._vol is None and self._vol_path is not None:
                # Load the origin of the bounding box & the voxel spacing value
                bbox = self._pipeline_cfg["Voxels"].get("bounding_box", None)
                self._origin = (sorted(bbox["x"])[0], sorted(bbox["y"])[0], sorted(bbox["z"])[0])
                self._spacing = self._pipeline_cfg["Voxels"].get("voxel_size", 1.0)
                # Load the volume file
                logger.info(f"Loading the volume file at: {self._vol_path}...")
                vol = read_volume(self._vol_path)
                logger.info("Creating a PyVista object...")
                self._vol = pyvista_volume(vol, origin=self._origin, spacing=self._spacing)
            if self._vol is not None:
                self._vol_actor = self.plotter.add_mesh(self._vol)
                self._render_volume(self._voxel_colormap)
            self._voxel_cmap_combo.setEnabled(True)
            # Update the combo box to reflect the current colormap
        else:
            if self._vol_actor is not None:
                self.plotter.remove_actor(self._vol_actor)
                self._vol_actor = None
                self.plotter.render()
            self._voxel_cmap_combo.setEnabled(False)

    def _on_pcd_toggled(self, state):
        if state == Qt.CheckState.Checked.value:
            if self._pcd is None and self._pcd_path is not None:
                logger.info(f"Loading the point cloud file at: {self._pcd_path}...")
                pcd = read_point_cloud(self._pcd_path)
                logger.info("Creating a PyVista object...")
                self._pcd = pv.PolyData(np.asarray(pcd.points))
            if self._pcd is not None:
                self._render_point_cloud(self._pcd_color)
            self._pcd_color_btn.setEnabled(True)
            self._pcd_opacity_slider.setEnabled(True)
            self._pcd_point_spin.setEnabled(True)
        else:
            if self._pcd_actor is not None:
                self.plotter.remove_actor(self._pcd_actor)
                self._pcd_actor = None
                self.plotter.render()
            self._pcd_color_btn.setEnabled(False)
            self._pcd_opacity_slider.setEnabled(False)
            self._pcd_point_spin.setEnabled(False)

    def _on_mesh_toggled(self, state):
        if state == Qt.CheckState.Checked.value:
            if self._mesh is None and self._mesh_path is not None:
                logger.info(f"Loading the mesh file at: {self._mesh_path}...")
                mesh = read_triangle_mesh(self._mesh_path)
                logger.info("Creating a PyVista object...")
                self._mesh = pyvista_mesh(mesh)
            if self._mesh is not None:
                self._render_mesh(self._mesh_color)
            self._mesh_color_btn.setEnabled(True)
        else:
            if self._mesh_actor is not None:
                self.plotter.remove_actor(self._mesh_actor)
                self._mesh_actor = None
                self.plotter.render()
            self._mesh_color_btn.setEnabled(True)

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

    def _on_voxel_cmap_changed(self, cmap_name: str):
        self._voxel_colormap = cmap_name
        self._apply_vol_color(cmap_name)

    def _on_pick_pcd_color(self):
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            hex_color = color.name()  # e.g. "#ff8800"
            self._pcd_color = hex_color
            self._apply_pcd_color(hex_color)

    def _on_pcd_opacity_changed(self, value: int):
        """Slider gives 0‑100 → map to 0.0‑1.0."""
        opacity = value / 100.0
        self._pcd_opacity = opacity
        if self._pcd_actor is not None:
            self._pcd_actor.GetProperty().SetOpacity(opacity)
            self.plotter.render()

    def _on_pcd_point_size_changed(self, size: int):
        """Spin box directly gives the desired point size."""
        self._pcd_point_size = size
        if self._pcd_actor is not None:
            self._pcd_actor.GetProperty().SetPointSize(size)
            self.plotter.render()

    def _on_pick_mesh_color(self):
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            hex_color = color.name()  # e.g. "#ff8800"
            self._mesh_color = hex_color
            self._apply_mesh_color(hex_color)

    def _on_grid_toggled(self, state):
        if state == Qt.CheckState.Checked.value:
            self._render_grid()
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

        # Reset lazy loading path references
        self._vol = None
        self._pcd = None
        self._mesh = None

        # Voxels - store file path for lazy loading
        try:
            vol_fs = scan.get_fileset(fs_matches['Voxels'])
            self._vol_path = vol_fs.get_file('Voxels').path()
            logger.info(f"Found the 'Voxels' associated to scan dataset '{scan_id}'...")
        except (FilesetNotFoundError, KeyError):
            logger.error(f"Could not find a 'Voxels' associated to scan dataset '{scan_id}'!")
            self._vol_path = None
        self._voxel_checkbox.setEnabled(self._vol_path is not None)
        self._voxel_checkbox.blockSignals(True)
        self._voxel_checkbox.setChecked(False)
        self._voxel_checkbox.blockSignals(False)

        # Point cloud - store file path for lazy loading
        try:
            pcd_fs = scan.get_fileset(fs_matches['PointCloud'])
            self._pcd_path = pcd_fs.get_file('PointCloud').path()
            logger.info(f"Found the 'PointCloud' associated to scan dataset '{scan_id}'...")
        except (FilesetNotFoundError, KeyError):
            logger.error(f"Could not find a 'PointCloud' associated to scan dataset '{scan_id}'!")
            self._pcd_path = None
        self._pcd_checkbox.setEnabled(self._pcd_path is not None)
        self._pcd_checkbox.blockSignals(True)
        self._pcd_checkbox.setChecked(False)
        self._pcd_checkbox.blockSignals(False)

        # Triangular Mesh - store file path for lazy loading
        try:
            mesh_fs = scan.get_fileset(fs_matches['TriangleMesh'])
            self._mesh_path = mesh_fs.get_file('TriangleMesh').path()
            logger.info(f"Found the 'TriangleMesh' associated to scan dataset '{scan_id}'...")
        except (FilesetNotFoundError, KeyError):
            logger.error(f"Could not find a 'TriangleMesh' associated to scan dataset '{scan_id}'!")
            self._mesh_path = None
        self._mesh_checkbox.setEnabled(self._mesh_path is not None)
        self._mesh_checkbox.blockSignals(True)
        self._mesh_checkbox.setChecked(False)
        self._mesh_checkbox.blockSignals(False)

        # Images fileset
        self._images_fs = scan.get_fileset('images')
        self._image_files = self._images_fs.get_files()

        # Reset plotter
        self.plotter.clear()
        # Reset actors & world parameters
        self._vol_actor = None
        self._pcd_actor = None
        self._mesh_actor = None
        self._origin = None
        self._spacing = None

        # Load the reconstruction pipeline configuration
        self._pipeline_cfg = toml.load(scan.path() / "pipeline.toml")

        # Setup slider
        n = len(self._image_files)
        self._image_slider.blockSignals(True)
        self._image_slider.setMinimum(0)
        self._image_slider.setMaximum(max(0, n - 1))
        # Keep the tick interval in sync with the new range
        self._image_slider.setTickInterval(1 if n > 1 else 0)
        self._image_slider.setValue(0)
        self._image_slider.setEnabled(n > 0)
        self._image_slider.blockSignals(False)

        if n > 0:
            # Call the same slot used when the user moves the slider.
            # This updates the label, background image, camera, etc.
            self._on_slider_changed(0)

        # Enable controls
        self._reset_cam_btn.setEnabled(True)
        self._snapshot_btn.setEnabled(True)


    def _render_volume(self, colormap):
        """Add (or replace) the point cloud actor."""
        if self._vol_actor is not None:
            self.plotter.remove_actor(self._vol_actor)
            self._vol_actor = None

        scalar_bar_args = dict(
            vertical=True,  # make the bar vertical
            color='white',  # title & tick labels in white
            title_font_size=20,
            label_font_size=16,
            fmt='{0:.1f}',
        )

        common = dict(reset_camera=False)
        self._vol_actor = self.plotter.add_volume(self._vol, cmap=colormap, scalar_bar_args=scalar_bar_args, **common)
        self._render_grid()
        self.plotter.render()

    def _apply_vol_color(self, colormap):
        self._render_volume(colormap)

    def _render_point_cloud(self, color):
        """Add (or replace) the point cloud actor."""
        if self._pcd_actor is not None:
            self.plotter.remove_actor(self._pcd_actor)
            self._pcd_actor = None

        common = dict(point_size=self._pcd_point_size, render_points_as_spheres=False, style="points",
                      opacity=self._pcd_opacity, reset_camera=False)
        self._pcd_actor = self.plotter.add_mesh(self._pcd, color=color, **common)
        self._render_grid()
        self.plotter.render()

    def _apply_pcd_color(self, color):
        self._render_point_cloud(color)

    def _render_mesh(self, color):
        """Add (or replace) the triangular mesh actor."""
        if self._mesh_actor is not None:
            self.plotter.remove_actor(self._mesh_actor)
            self._mesh_actor = None

        common = dict(style="surface", lighting=True, reset_camera=False)
        self._mesh_actor = self.plotter.add_mesh(self._mesh, color=color, **common)
        self._render_grid()
        self.plotter.render()

    def _apply_mesh_color(self, color):
        self._render_mesh(color)

    def _render_grid(self):
        if self._vol_actor is not None:
            bounds = self._vol.bounds
            # PyVista ImageData provides .bounds as (xmin, xmax, ymin, ymax, zmin, zmax)
            self.plotter.show_grid(color='white', bounds=bounds)
        elif self._pcd_actor is not None:
            # PyVista PolyData provides .bounds as (xmin, xmax, ymin, ymax, zmin, zmax)
            bounds = self._pcd.bounds
            self.plotter.show_grid(color='white', bounds=bounds)

    def _apply_image(self, image_f):
        """Update the background image and camera to match the selected image."""
        try:
            cam_pos, focal_point, up_world, fov_y_deg = _camera_params_from_file(image_f)
        except Exception:
            # Metadata missing - just show the image, skip camera update
            cam_pos = focal_point = up_world = fov_y_deg = None

        try:
            self.plotter.add_background_image(image_f.path(), as_global=False)
        except:
            self.plotter.remove_background_image()
            self.plotter.add_background_image(image_f.path(), as_global=False)

        self.plotter.add_text(
            f"Image: {image_f.id}",
            position="upper_edge",
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
