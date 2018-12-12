import numpy as np
from scipy.ndimage import binary_opening, binary_closing
import matplotlib.pyplot as plt

def compute_mask_TGI(threshold=2.0):
    image = np.asarray(image, dtype=float)
    TGI = image[:,:,1] - 0.39*image[:,:,0] - 0.61*image[:,:,2]
    return TGI > threshold
