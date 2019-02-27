import FileIO
from skimage.filters import gaussian
import open3d
from lettucescan.pcd import *
from open3d.geometry import *
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.special import betainc
from lettucescan.cl import *

model = "Isotropic3"
FileHFM_executable = "FileHFM_"+model
FileHFM_binary_dir = "/home/twintz/.local/bin"
eps = 1e-3


def get_roots(pts, tol=0, axis=2, inverted=True):
    if inverted:
        val = np.max(pts[:, axis])
        res = pts[pts[:, axis] >= val - tol, :]
    else:
        val = np.min(pts[:, axis])
        res = pts[pts[:, axis] <= val + tol, :]
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


def hfm_compute_flow(speed, seeds, origin, voxel_size):
    input = {
        "arrayOrdering": "RowMajor",
        "order": 2,
        "model": model,
        "dims": np.array(speed.shape),
        "gridScale": voxel_size,
        "origin": origin,
        "seeds": seeds,
        "speed": speed,
        "exportValues": 1,
        "exportGeodesicFlow": 1
    }
    result = FileIO.WriteCallRead(input,
                                  FileHFM_executable,
                                  binary_dir=FileHFM_binary_dir,
                                  logFile=None)  # Display output directly

    return result['values'], result['geodesicFlow']


if __name__ == "__main__":
    pcd = read_point_cloud("../data/2019-02-01_11-16-35/3d/voxels.ply")
    pts = np.asarray(pcd.points)
    voxel_size = 1.0
    volume, origin = pcd2vol(pts, voxel_size)
    maxval = np.max(pts, axis=0)
    seeds = get_roots(pts, 2*voxel_size)
    speed = vesselness_3D(volume, 1)
    values, flow = hfm_compute_flow(speed, seeds, origin, voxel_size)

    x = Geodesics()

    N_geodesics = 100000
    tips = origin[np.newaxis, :] + np.random.rand(N_geodesics, 3) * (
        maxval - origin)[np.newaxis, :]
    votes = x.compute_geodesics(values, origin, voxel_size, flow, tips, 10000, 0.25)

    tipsidx = point2index(tips, origin, voxel_size) 
