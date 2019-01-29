import numpy as np
from abc import ABC, abstractmethod
from scipy.ndimage import binary_opening, binary_closing

def MaskingBlockParameters:
    def __init__(self, f)
        self.f = f

def LinearMasking(MaskingBlockParameters):
    def __init__(self, coefs):
        self.f = lambda x: (coefs[0] * x[:,:,0] + coefs[1] * x[:,:,1] +
                coefs[2] * x[:,:,2]) > coefs[3]

eps = 1e-9
def excess_green(x):
    s = x.sum(axis=2) + eps
    r = x[:,:,0] / s
    g = x[:,:,1] / s
    b = x[:,:,2] / s
    return (2*g - r - b)

def ExcessGreenMasking(MaskingBlockParameters):
    def __init__(self, threshold):
        self.f = lambda x: excess_green(x) > threshold

def MaskingBlock(ProcessingBlock):
    def __init__(self, input_filesets, output_filesets, params):
        super().__init__(input_filesets, output_filesets, params)
        self.f = self.params.f

    def get_input_filesets(self):
        return ['images']

    def get_output_filesets(self):
        return ['masks']

    def process(self):
        for i,file in enumerate(input_filesets['images'].get_files()):
            im = file.read_image()
            im = np.asarray(im, dtype=float) / 255.0
            mask = (uint8) (self.f(im) * 255)
            mask_file = output_filesets['masks'].create_file(file.id)
            mask_file.write_image("png", mask)
