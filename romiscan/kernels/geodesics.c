#include "common.h"

__kernel void geodesic(__read_only image3d_t gx,
                    __read_only image3d_t gy,
                    __read_only image3d_t gz,
                    __read_only image3d_t values,
                    __global int* votes,
                    __global float* points,
                    __global uchar* labels,
                    __global int* points_remain,
                    __global int* shape,
                    float step_size)
{

    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_NONE |
                            CLK_FILTER_LINEAR;

    atomic_and(points_remain, 0);

    int i = get_global_id(0);

    int nx = shape[0];
    int ny = shape[1];
    int nz = shape[2];


    if (!labels[i])
    {
        return;
    }

    float x = points[3*i];
    float y = points[3*i+1];
    float z = points[3*i+2];
    float4 pt = {z, y, x, 0.0f};


    float val = read_imagef(values, smp, pt).x;

    float gx_val = read_imagef(gx, smp, pt).x;
    float gy_val = read_imagef(gy, smp, pt).x;
    float gz_val = read_imagef(gz, smp, pt).x;

    x -=  gx_val * step_size;
    y -=  gy_val * step_size;
    z -=  gz_val * step_size;

    float4 new_pt = {z, y, x, 0.0f};
    float new_val = read_imagef(values, smp, new_pt).x;

    points[3*i] = x;
    points[3*i+1] = y;
    points[3*i+2] = z;

    if (new_val < step_size || new_val >= val) {
        labels[i] = 0;
        return;
    }

    atomic_or(points_remain, 1);

    int xi = x;
    int yi = y;
    int zi = z;
    if(xi >= 0 && xi < nx && yi >= 0 && yi < ny && zi >= 0 && zi < nz) {
        int idx = xi * ny * nz + yi * nz + zi;
        atomic_add(&votes[idx], 1);
    }
    return;
}

