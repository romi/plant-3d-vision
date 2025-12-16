#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import open3d as o3d
from PyQt6.QtCore import QThread
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtWidgets import QProgressBar
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from plant3dvision.proc3d import pcd2mesh


class TriangulationThread(QThread):
    finished = pyqtSignal(bool, str)
    meshReady = pyqtSignal(object)
    progress = pyqtSignal(str)  # New signal for progress updates

    def __init__(self, point_cloud_path):
        super().__init__()
        self.point_cloud_path = point_cloud_path

    def run(self):
        try:
            self.progress.emit("Loading point cloud...")
            pcd = o3d.io.read_point_cloud(self.point_cloud_path)

            self.progress.emit("Creating mesh...")
            mesh = pcd2mesh(pcd)

            self.progress.emit("Finalizing...")
            self.meshReady.emit(mesh)
            self.finished.emit(True, "Triangulation completed!")
        except Exception as e:
            self.finished.emit(False, f"An error occurred during triangulation: {str(e)}")


class PointCloudApp(QMainWindow):
    def __init__(self, app_instance):
        super().__init__()

        self.app = app_instance
        self.point_cloud_path = None
        self.mesh = None
        self.thread = None  # Keep track of the thread

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Point Cloud Triangulation")
        self.setGeometry(100, 100, 800, 600)
        self.setAcceptDrops(True)

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.layout = QVBoxLayout()
        self.centralWidget.setLayout(self.layout)

        self.label = QLabel("Drag and drop a PLY point cloud file here", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label)

        self.triangulateButton = QPushButton("Triangulate", self)
        self.triangulateButton.clicked.connect(self.startTriangulation)
        self.layout.addWidget(self.triangulateButton)

        # Progress bar to show busy indication
        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0, 0)  # Indeterminate mode
        self.progressBar.setTextVisible(False)
        self.layout.addWidget(self.progressBar)
        self.progressBar.hide()

        self.saveButton = QPushButton("Save Mesh as PLY", self)
        self.saveButton.clicked.connect(self.saveMesh)
        self.layout.addWidget(self.saveButton)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            # Call the base class implementation for any other case
            super(PointCloudApp, self).dragEnterEvent(event)

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            self.point_cloud_path = file_path
            self.label.setText(f"Dropped file: {file_path}")
            print("Point cloud file dropped:", file_path)

        # Call the base class implementation to handle other cases
        super(PointCloudApp, self).dropEvent(event)

    def startTriangulation(self):
        if not self.point_cloud_path:
            QMessageBox.warning(self, "Warning", "No point cloud file selected.")
            return

        # Show progress bar and disable buttons during triangulation
        self.progressBar.show()
        self.triangulateButton.setEnabled(False)
        self.saveButton.setEnabled(False)
        self.label.setText("Processing...")

        # Create and configure the thread
        self.thread = TriangulationThread(self.point_cloud_path)
        self.thread.finished.connect(self.onTriangulationFinished)
        self.thread.meshReady.connect(self.setMesh)
        self.thread.progress.connect(self.updateProgress)  # Connect progress signal

        # Start the thread
        self.thread.start()

    def updateProgress(self, message):
        """Update the UI with progress information"""
        self.label.setText(message)

    def setMesh(self, mesh):
        self.mesh = mesh

    def onTriangulationFinished(self, success, message):
        # Hide progress bar and re-enable buttons after triangulation is done
        self.progressBar.hide()
        self.triangulateButton.setEnabled(True)
        self.saveButton.setEnabled(True)

        if success:
            self.label.setText("Triangulation completed!")
        else:
            self.label.setText("Error during triangulation")

        # Clean up the thread
        if self.thread is not None:
            self.thread.deleteLater()
            self.thread = None

    def saveMesh(self):
        if not self.mesh:
            QMessageBox.warning(self, "Warning", "No mesh to save. Please triangulate first.")
            return

        # Use the flags directly in the function call instead of Options()
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mesh as PLY",
            "",
            "PLY Files (*.ply);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if fileName:
            o3d.io.write_triangle_mesh(fileName, self.mesh)
            QMessageBox.information(self, "Success", f"Mesh saved successfully to {fileName}")


def main():
    app = QApplication(sys.argv)
    ex = PointCloudApp(app)  # Pass the app instance to PointCloudApp
    ex.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
