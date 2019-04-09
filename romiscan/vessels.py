from skimage.filters import gaussian
from romiscan.pcd import *
from open3d.open3d.geometry import *
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.special import betainc
from romiscan.cl import *

eps = 1e-3


def get_roots(pts, tol=0, axis=2, inverted=True):
    if inverted:
        val = np.max(pts[:, axis])
        res = pts[pts[:, axis] >= val - tol, :]
    else:
        val = np.min(pts[:, axis])
        res = pts[pts[:, axis] <= val + tol, :]
    print(res)
    return res


def hessian_eigvals_abs_sorted(volume):
    N = volume.ndim
    H = hessian_matrix(volume)
    L = hessian_matrix_eigvals(H)

    sorting = np.argsort(np.abs(L), axis=0)

    res = []
    for i in range(N):
        newL = np.zeros_like(L[0])
        for j in range(N):
            newL[sorting[i, :] == j] = L[j, sorting[i, :] == j]
        res.append(newL)
    return res


def vesselness_3D(volume, scale):
    volume = gaussian(volume, scale)
    L1, L2, L3 = hessian_eigvals_abs_sorted(volume)
    S = np.sqrt(L1**2 + L2**2 + L3**2)
    # res = betainc(5, 8, L2 / (L3 + eps)) * (1 - betainc(4, 4, np.abs(L1/(L2 + eps))))
    res = np.sqrt(L3**2 - (L3-L2)**2 - L1**2)  # / (1+S + eps)
    res[np.isnan(res)] = 0

    res = res / res.max()
    res = res + 0.01
    return res

def vesselness_2D(image, scale):
    image = image.astype(float)
    image = gaussian(image, scale)
    L1,L2 = hessian_eigvals_abs_sorted(image)
    res = (1 - betainc(4, 4, np.abs(L1)/(np.abs(L2) + eps)))
    res = np.abs(L2)/np.abs(L2).max()*np.exp(np.abs(L1)/(np.abs(L2) + eps))
    return res
