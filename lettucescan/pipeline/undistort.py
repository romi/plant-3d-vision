from imageio import imread
from lettucethink import fsdb
from scanner import localdirs

import cv2
import numpy as np
from scipy.ndimage import binary_opening, binary_closing

from lettucescan.pipeline.processing_block import ProcessingBlock

eps = 1e-9


class Undistort(ProcessingBlock):
    def read_input(self, scan, endpoint):
        fileset = scan.get_fileset(endpoint)

        if self.camera_model is None:
            scanner_metadata = scan.get_metadata('scanner')
            self.camera = scanner_metadata['camera_model']
        else:
            self.camera = camera_model

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

    def __init__(self, camera_model=None):
        self.camera_model = camera_model

    def process(self):
        camera_model = self.camera['parameters']

        self.undistorted_images = []

        for img in self.images:
            data = img['data']
            mat = np.matrix([[camera_model['fx'], 0, camera_model['cx']],
                             [0, camera_model['fy'], camera_model['cy']],
                             [0, 0, 1]])
            undistort_parameters = np.array([camera_model['k1'], camera_model['k2'],
                                             camera_model['p1'],
                                             camera_model['p2']])
            undistorted_data = cv2.undistort(im, mat, undistort_parameters)
            self.undistorted_images.append({
                'id': img['id'],
                'data': undistorted_data,
                'metadata': img['metadata']
            })
