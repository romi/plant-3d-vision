#include "common.h"

bool backproject_point(float3 pt, __global float *intrinsics,
                       __global float *rot, __global float *tvec,
                       __read_only image2d_t mask, int2 *res) {
    float f_x = intrinsics[0];
    float f_y = intrinsics[1];
    float c_x = intrinsics[2];
    float c_y = intrinsics[3];

    float p_z = rot[6] * pt.x + rot[7] * pt.y + rot[8] * pt.z + tvec[2];

    if (p_z < 0) {
        return false;
    }

    float p_x = rot[0] * pt.x + rot[1] * pt.y + rot[2] * pt.z + tvec[0];
    float p_y = rot[3] * pt.x + rot[4] * pt.y + rot[5] * pt.z + tvec[1];

    p_x = p_x / p_z * f_x + c_x;
    p_y = p_y / p_z * f_y + c_y;

    res->x = (int)p_x;
    res->y = (int)p_y;

    if (res->x < 0 || res->x > get_image_width(mask) - 1) {
        return false;
    }
    if (res->y < 0 || res->y > get_image_height(mask) - 1) {
        return false;
    }

    return true;
}

__kernel void average(__read_only image2d_t mask, __global float *value,
                             __global float *intrinsics, __global float *rot,
                             __global float *tvec, __global float *volinfo,
                             __global int *shape) {
    const sampler_t samplerA =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR;
    int idx = get_global_id(0);
    int3 ijk = unravel_index(idx, shape);
    float3 pt = {volinfo[0] + ijk.x * volinfo[3],
                 volinfo[1] + ijk.y * volinfo[3],
                 volinfo[2] + ijk.z * volinfo[3]};

    int2 imagept;
    if (!backproject_point(pt, intrinsics, rot, tvec, mask, &imagept)) {
        return;
    }
    value[idx] += read_imagef(mask, samplerA, imagept).x;
}

__kernel void carve(__read_only image2d_t mask, __global int *labels,
                    __global float *intrinsics, __global float *rot,
                    __global float *tvec, __global float *volinfo,
                    __global int *shape) {
    const sampler_t samplerA =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

    int idx = get_global_id(0);
    int3 ijk = unravel_index(idx, shape);

    if (labels[idx] == -1) {
        return;
    }

    float3 pt = {volinfo[0] + ijk.x * volinfo[3],
                 volinfo[1] + ijk.y * volinfo[3],
                 volinfo[2] + ijk.z * volinfo[3]};

    int2 imagept;
    if (!backproject_point(pt, intrinsics, rot, tvec, mask, &imagept)) {
        return;
    }
    if (read_imagei(mask, samplerA, imagept).x == 0) {
        labels[idx] = -1;
    } else if (labels[idx] == 0) { // Mark a voxel the first time it is seen
        labels[idx] = 1;
    }
}
