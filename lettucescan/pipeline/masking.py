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
    def __init__(self, input_fileset, output_fileset, f):
        self.f = f
        self.input_fileset = input_fileset
        self.output_fileset = output_fileset

    def process(self):
        for i, file in enumerate(self.input_fileset.get_files()):
            im = file.read_image()
            im = np.asarray(im, dtype=float) / 255.0
            mask = np.asarray((self.f(im) * 255), dtype=np.uint8)
            mask_file = self.output_fileset.create_file(file.id)
            mask_file.write_image('png', mask)


class ExcessGreenMasking(Masking):
    def __init__(self, input_filesets, output_filesets, threshold):
        def f(x): return excess_green(x) > threshold
        super().__init__(input_filesets, output_filesets, f)


class LinearMasking(Masking):
    def __init__(self, input_filesets, output_filesets, coefs):
        def f(x): return (coefs[0] * x[:, :, 0] + coefs[1] * x[:, :, 1] +
                          coefs[2] * x[:, :, 2]) > coefs[3]
        super().__init__(input_filesets, output_filesets, f)
