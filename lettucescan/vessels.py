import FileIO
from skimage.filters import gaussian
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

model = "Isotropic3"
FileHFM_executable = "FileHFM_"+model

def get_hessian_eigvals_abs_sorted(volume):
    H = hessian_matrix(volume)
    L = hessian_matrix_eigvals(H)

def vesselness_filter(volume, scale):
    volume = gaussian(volume, scale)

def hfm_compute_flow(speed, seeds, voxel_size):
    input = {
    "arrayOrdering": "RowMajor",
    "model": model,
    "dims": np.array(speed.shape),
    "gridScale": voxel_size,
    "seeds": seeds,
    "speed" : speed,
    "exportValues": 1,
    "exportGeodesicFlow": 1,

    "speed":np.roll(boatSpeed,int(round(0.3*nTheta)),axis=0),

    "uniformlySampledSpeed":nTheta,
}
