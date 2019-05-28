import numpy as np
from scipy.special import betainc
from skimage.filters import gaussian
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

EPS = 1e-9

def excess_green(x):
    s = x.sum(axis=2) + EPS
    r = x[:, :, 0] / s
    g = x[:, :, 1] / s
    b = x[:, :, 2] / s
    return (2*g - r - b)

def hessian_eigvals_abs_sorted(volume):
    N = volume.ndim
    H = hessian_matrix(volume, order="xy")
    L = hessian_matrix_eigvals(H)

    sorting = np.argsort(np.abs(L), axis=0)

    res = []
    for i in range(N):
        newL = np.zeros_like(L[0])
        for j in range(N):
            newL[sorting[i, :] == j] = L[j, sorting[i, :] == j]
        res.append(newL)
    return res

def vesselness(image, scale):
    image = image.astype(float)
    image = gaussian(image, scale)
    L1, L2 = hessian_eigvals_abs_sorted(image)
    res = (1 - betainc(4, 4, np.abs(L1) / (np.abs(L2) + EPS)))
    res = np.abs(L2) / np.abs(L2).max() * np.exp(
        np.abs(L1) / (np.abs(L2) + EPS))
    return res
