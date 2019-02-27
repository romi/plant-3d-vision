import numpy as np
from scipy.ndimage import binary_opening, binary_closing, binary_dilation

eps = 1e-9

def excess_green(x):
    s = x.sum(axis=2) + eps
    r = x[:, :, 0] / s
    g = x[:, :, 1] / s
    b = x[:, :, 2] / s
    return (2*g - r - b)


