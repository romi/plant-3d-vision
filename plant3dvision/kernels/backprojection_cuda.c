// CUDA includes
#include <cuda_runtime.h>

// Utility for 3D unravel (flattened to 3D index)
// This matches the OpenCL common.h indexing convention
__device__ void unravel_index(int idx, int3 shape, int &x, int &y, int &z) {
    // int nx = shape.x;  // x dimension
    int ny = shape.y;  // y dimension
    int nz = shape.z;  // z dimension

    // Calculate coordinates in the same order as OpenCL version
    x = idx / (ny * nz);                   // i coordinate (x)
    y = (idx - x * ny * nz) / nz;          // j coordinate (y)
    z = idx - x * ny * nz - y * nz;        // k coordinate (z)
}

// Core backprojection function
__device__ bool backproject_point(
    float3 pt, const float *intrinsics,
    const float *rot, const float *tvec,
    int width, int height,
    int &out_x, int &out_y)
{
    float f_x = intrinsics[0], f_y = intrinsics[1];
    float c_x = intrinsics[2], c_y = intrinsics[3];

    // Transform point to camera coordinates and project
    float p_z = rot[6]*pt.x + rot[7]*pt.y + rot[8]*pt.z + tvec[2];
    if (p_z <= 0) return false;

    float p_x = rot[0]*pt.x + rot[1]*pt.y + rot[2]*pt.z + tvec[0];
    float p_y = rot[3]*pt.x + rot[4]*pt.y + rot[5]*pt.z + tvec[1];

    // Apply intrinsics
    p_x = p_x / p_z * f_x + c_x;
    p_y = p_y / p_z * f_y + c_y;

    out_x = (int)p_x;
    out_y = (int)p_y;

    return (out_x >= 0 && out_x < width && out_y >= 0 && out_y < height);
}

extern "C" __global__
void average_kernel(
    const float *mask, // mask: width x height (row-major float32)
    float *value,
    const float *intrinsics,
    const float *rot,
    const float *tvec,
    const float *volinfo,
    const int *shape,
    int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_x = shape[0], dim_y = shape[1], dim_z = shape[2];
    if (idx >= dim_x * dim_y * dim_z) return;

    int x, y, z;
    unravel_index(idx, make_int3(dim_x, dim_y, dim_z), x, y, z);

    // Compute 3D point in world coordinates
    float3 pt = make_float3(
        volinfo[0] + x * volinfo[3],  // origin_x + x * voxel_size
        volinfo[1] + y * volinfo[3],  // origin_y + y * voxel_size
        volinfo[2] + z * volinfo[3]   // origin_z + z * voxel_size
    );

    int px, py;
    if (!backproject_point(pt, intrinsics, rot, tvec, width, height, px, py))
        return;

    // Safely access the mask and accumulate
    float mask_val = mask[py * width + px];
    atomicAdd(&value[idx], mask_val);
}

extern "C" __global__
void carve_kernel(
    const float *mask, // mask as float data (0 or 1)
    int *labels,
    const float *intrinsics,
    const float *rot,
    const float *tvec,
    const float *volinfo,
    const int *shape,
    int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_x = shape[0], dim_y = shape[1], dim_z = shape[2];
    if (idx >= dim_x * dim_y * dim_z) return;

    if (labels[idx] == -1) return;

    int x, y, z;
    unravel_index(idx, make_int3(dim_x, dim_y, dim_z), x, y, z);

    // Compute 3D point in world coordinates
    float3 pt = make_float3(
        volinfo[0] + x * volinfo[3],  // origin_x + x * voxel_size
        volinfo[1] + y * volinfo[3],  // origin_y + y * voxel_size
        volinfo[2] + z * volinfo[3]   // origin_z + z * voxel_size
    );

    int px, py;
    if (!backproject_point(pt, intrinsics, rot, tvec, width, height, px, py))
        return;

    float mval = mask[py * width + px];
    if (mval == 0.0f)
        labels[idx] = -1;
    else if (labels[idx] == 0)
        labels[idx] = 1;
}

// Bayesian space carving (log-odds) vote for a single view.
// The `value` buffer is pre-initialized to the prior log-odds; each view adds
// log(tpr/fpr) where the voxel projects inside the mask, or
// log((1-tpr)/(1-fpr)) where it projects on background. Mirrors the Julia
// ROMIVoxels.jl `voxel_voting`.
extern "C" __global__
void bayes_kernel(
    const float *mask, // mask: width x height (row-major float32, 0/1)
    float *value,      // log-odds volume (float32), accumulates across views
    const float *intrinsics,
    const float *rot,
    const float *tvec,
    const float *volinfo,
    const int *shape,
    int width, int height,
    float log_occ,     // log(tpr / fpr)
    float log_empty)   // log((1 - tpr) / (1 - fpr))
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_x = shape[0], dim_y = shape[1], dim_z = shape[2];
    if (idx >= dim_x * dim_y * dim_z) return;

    int x, y, z;
    unravel_index(idx, make_int3(dim_x, dim_y, dim_z), x, y, z);

    // Compute 3D point in world coordinates
    float3 pt = make_float3(
        volinfo[0] + x * volinfo[3],  // origin_x + x * voxel_size
        volinfo[1] + y * volinfo[3],  // origin_y + y * voxel_size
        volinfo[2] + z * volinfo[3]   // origin_z + z * voxel_size
    );

    int px, py;
    if (!backproject_point(pt, intrinsics, rot, tvec, width, height, px, py))
        return;

    // Background projection (mask == 0) also contributes a (negative) vote.
    float mval = mask[py * width + px];
    float vote = (mval > 0.5f) ? log_occ : log_empty;
    atomicAdd(&value[idx], vote);
}
