import numpy as np
from scipy.ndimage import binary_opening, binary_closing

from lettucescan.pipeline.processing_block import ProcessingBlock


eps = 1e-9


def excess_green(x):
    s = x.sum(axis=2) + eps
    r = x[:, :, 0] / s
    g = x[:, :, 1] / s
    b = x[:, :, 2] / s
    return (2*g - r - b)


class Masking(ProcessingBlock):
    def read_input(self, scan, endpoint):
        fileset = scan.get_fileset(endpoint)

        scanner_metadata = scan.get_metadata('scanner')
        self.camera = scanner_metadata['camera_model']

        self.images = []
        for f in fileset.get_files():
            data = f.read_image()
            self.images.append({
                'id': f.id,
                'data': data,
                'metadata': f.get_metadata()
            })

    def write_output(self, scan, endpoint):
        fileset = scan.get_fileset(endpoint)
        for img in self.masks:
            f = fileset.get_file(img['id'])
            if f is None:
                f = fileset.create_file(img['id'])
                f.write_image('png', img['data'])
                f.set_metadata(img['metadata'])


    def __init__(self, input_fileset, output_fileset, f):
        self.f = f

    def process(self):
        self.masks = []
        for img in self.images:
            im = img['data']
            im = np.asarray(im, dtype=float) / 255.0
            mask_data = np.asarray((self.f(im) * 255), dtype=np.uint8)
            self.undistorted_images.append({
                'id': img['id'],
                'data': mask_data,
                'metadata': img['metadata']
            })


class ExcessGreenMasking(Masking):
    def __init__(self, threshold):
        def f(x): return excess_green(x) > threshold
        super().__init__(f)


class LinearMasking(Masking):
    def __init__(self, coefs):
        def f(x): return (coefs[0] * x[:, :, 0] + coefs[1] * x[:, :, 1] +
                          coefs[2] * x[:, :, 2]) > coefs[3]
        super().__init__(f)
