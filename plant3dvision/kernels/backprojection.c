/**
 * @file backprojection.c
 * @brief This file contains functions and OpenCL kernels for projecting 3D points 
 *        into 2D image space and updating voxel labels or values based on a mask image.
 */

#include "common.h"

/**
 * @brief Projects a 3D point onto the 2D image plane using given camera intrinsics,
 *        rotation matrix, and translation vector.
 *
 * @param pt            The 3D point to be projected.
 * @param intrinsics    Pointer to camera intrinsic parameters: [f_x, f_y, c_x, c_y].
 * @param rot           Pointer to the 3x3 rotation matrix in row-major order.
 * @param tvec          Pointer to the translation vector: [tx, ty, tz].
 * @param mask          The image2d_t mask used to check the valid projection range.
 * @param res           Pointer to the output 2D coordinate (pixel location).
 * @return              true if point projects into the valid image region, false otherwise.
 */
bool backproject_point(float3 pt, __global float *intrinsics,
                       __global float *rot, __global float *tvec,
                       __read_only image2d_t mask, int2 *res) 
{
    // Extract intrinsic parameters
    float f_x = intrinsics[0];
    float f_y = intrinsics[1];
    float c_x = intrinsics[2];
    float c_y = intrinsics[3];

    // Compute depth in the rotated coordinate system
    float p_z = rot[6] * pt.x + rot[7] * pt.y + rot[8] * pt.z + tvec[2];

    // If the depth is behind the camera or non-positive, reject the point
    if (p_z < 0) {
        return false;
    }

    // Compute x and y in the camera coordinate system
    float p_x = rot[0] * pt.x + rot[1] * pt.y + rot[2] * pt.z + tvec[0];
    float p_y = rot[3] * pt.x + rot[4] * pt.y + rot[5] * pt.z + tvec[1];

    // Normalize by depth and apply intrinsics (focal length, principal point)
    p_x = p_x / p_z * f_x + c_x;
    p_y = p_y / p_z * f_y + c_y;

    // Convert to integer pixel indices
    res->x = (int)p_x;
    res->y = (int)p_y;

    // Check if the projected pixel is within the mask image boundaries
    if (res->x < 0 || res->x > get_image_width(mask) - 1) {
        return false;
    }
    if (res->y < 0 || res->y > get_image_height(mask) - 1) {
        return false;
    }

    return true;
}

/**
 * @brief An OpenCL kernel that calculates a running sum (average) of mask values 
 *        for each voxel in a 3D volume by projecting the voxel center onto the image plane.
 *
 * @param mask         The image2d_t mask used for sampling.
 * @param value        Global array storing the accumulated mask values for each voxel.
 * @param intrinsics   Pointer to camera intrinsic parameters.
 * @param rot          Pointer to rotation matrix.
 * @param tvec         Pointer to translation vector.
 * @param volinfo      Pointer to volume coordinate data: 
 *                     [origin_x, origin_y, origin_z, voxel_size].
 * @param shape        Pointer to volume shape: [dim_x, dim_y, dim_z].
 */
__kernel void average(__read_only image2d_t mask, __global float *value,
                      __global float *intrinsics, __global float *rot,
                      __global float *tvec, __global float *volinfo,
                      __global int *shape) 
{
    // Sampler configuration. We disable normalized coords so we can use integer image coordinates.
    const sampler_t samplerA =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR;

    // Global ID corresponds to one voxel index in the volume
    int idx = get_global_id(0);

    // Convert linear index to 3D coordinates (i, j, k)
    int3 ijk = unravel_index(idx, shape);

    // Compute the 3D point (voxel center) in world (or volume) coordinates
    float3 pt = {
        volinfo[0] + ijk.x * volinfo[3],
        volinfo[1] + ijk.y * volinfo[3],
        volinfo[2] + ijk.z * volinfo[3]
    };

    // Container for projected pixel coordinates
    int2 imagept;

    // Backproject the voxel center onto the image plane; if invalid, skip
    if (!backproject_point(pt, intrinsics, rot, tvec, mask, &imagept)) {
        return;
    }

    // Accumulate the mask value at this pixel location
    // read_imagef returns a float4, 'x' is the intensity we need
    value[idx] += read_imagef(mask, samplerA, imagept).x;
}

/**
 * @brief An OpenCL kernel that labels or carves out voxels based on a mask image. 
 *        Projects each voxel onto the image plane and updates its label accordingly.
 *
 * @param mask         The image2d_t mask used for looking up valid region.
 * @param labels       Global array storing integer labels for each voxel.
 * @param intrinsics   Pointer to camera intrinsic parameters.
 * @param rot          Pointer to rotation matrix.
 * @param tvec         Pointer to translation vector.
 * @param volinfo      Pointer to volume coordinate data.
 * @param shape        Pointer to volume shape: [dim_x, dim_y, dim_z].
 */
__kernel void carve(__read_only image2d_t mask, __global int *labels,
                    __global float *intrinsics, __global float *rot,
                    __global float *tvec, __global float *volinfo,
                    __global int *shape) 
{
    // Sampler configuration. NEAREST filtering to pick exact mask values
    const sampler_t samplerA =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

    // Global ID corresponds to one voxel index in the volume
    int idx = get_global_id(0);

    // Convert linear index to 3D coordinates (i, j, k)
    int3 ijk = unravel_index(idx, shape);

    // If a voxel is already carved out, skip processing
    if (labels[idx] == -1) {
        return;
    }

    // Compute the 3D point (voxel center)
    float3 pt = {
        volinfo[0] + ijk.x * volinfo[3],
        volinfo[1] + ijk.y * volinfo[3],
        volinfo[2] + ijk.z * volinfo[3]
    };

    // Container for projected pixel coordinates
    int2 imagept;

    // Backproject the voxel center onto the image plane; if invalid, skip
    if (!backproject_point(pt, intrinsics, rot, tvec, mask, &imagept)) {
        return;
    }

    // If the mask pixel is zero, mark voxel as carved out (-1)
    if (read_imagei(mask, samplerA, imagept).x == 0) {
        labels[idx] = -1;
    }
    // If voxel was never labeled, label it for the first time
    else if (labels[idx] == 0) {
        labels[idx] = 1;
    }
}