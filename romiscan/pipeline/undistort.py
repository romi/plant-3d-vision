from imageio import imread
from lettucethink import fsdb

import cv2
import numpy as np
from scipy.ndimage import binary_opening, binary_closing

from lettucescan.pipeline.processing_block import ProcessingBlock

eps = 1e-9


class Undistort(ProcessingBlock):
    def read_input(self, scan, endpoint):
        fileset = scan.get_fileset(endpoint)

        if self.camera is None:
            scanner_metadata = scan.get_metadata('scanner')
            self.camera = scanner_metadata['camera_model']
        else:
            self.camera = self.camera

        self.images = []
        for f in fileset.get_files():
            data = f.read_image()
            self.images.append({
                'id': f.id,
                'data': data,
                'metadata': f.get_metadata()
            })

    def write_output(self, scan, endpoint):
        fileset = scan.get_fileset(endpoint, create=True)
        for img in self.undistorted_images:
            f = fileset.get_file(img['id'], create=True)
            f.write_image('jpg', img['data'])
            f.set_metadata(img['metadata'])

    def __init__(self, camera=None):
        self.camera = camera

    def process(self):
        self.undistorted_images = []

        for img in self.images:
            data = img['data']
            camera_params = self.camera['params']
            mat = np.matrix([[camera_params[0], 0, camera_params[2]],
                             [0, camera_params[1], camera_params[3]],
                             [0, 0, 1]])
            undistort_parameters = np.array(camera_params[4:])
            undistorted_data = cv2.undistort(data, mat, undistort_parameters)
            self.undistorted_images.append({
                'id': img['id'],
                'data': undistorted_data,
                'metadata': img['metadata']
            })
