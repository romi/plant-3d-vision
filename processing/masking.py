import numpy as np
from scipy.ndimage import binary_opening, binary_closing
import matplotlib.pyplot as plt

def compute_mask_TGI(image, threshold=2.0):
    image = np.asarray(image, dtype=float)
    TGI = image[:,:,1] - 0.39*image[:,:,0] - 0.61*image[:,:,2]
    return 255*np.asarray(TGI > threshold, dtype=np.uint8)

def compute_mask_custom(image, coefs):
    image = np.asarray(image, dtype=float)
    image = image / image.max()
    coefs = np.asarray(coefs, dtype=float)
    return 255*np.asarray(coefs[0] * image[:,:,0] + coefs[1] * image[:,:,1] +
                coefs[2] * image[:,:,2] > coefs[3], dtype=np.uint8)
