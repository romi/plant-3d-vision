__kernel void geodesic(__read_only image3d_t gx,
                    __read_only image3d_t gy,
                    __read_only image3d_t gz,
                    __read_only image3d_t values,
                    __global unsigned int* votes,
                    __global float* points,
                    __global uchar* labels,
                    __global int* points_remain,
                    float step_size, int nx, int ny, int nz)
{

    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_NONE |
                            CLK_FILTER_LINEAR;

    int i = get_global_id(0);
    if (!labels[i])
    {
        return;
    }
    float4 pt = {points[3*i], points[3*i+1], points[3*i+2], 0.0f};

    float val = read_imagef(values, smp, pt).x;

    float gx_val = read_imagef(gx, smp, pt).x;
    float gy_val = read_imagef(gy, smp, pt).x;
    float gz_val = read_imagef(gz, smp, pt).x;

    float4 new_pt = {pt.x - step_size*gx_val, pt.y - step_size*gy_val, pt.z - step_size*gz_val, 0.0f};
    float new_val = read_imagef(values, smp, new_pt).x;

    if (new_val < step_size || new_val >= val) {
        labels[i] = 0;
        return;
    }

    atomic_or(points_remain, 1);

    int idx = new_pt[0]*ny*nz + new_pt[1]*ny + new_pt[2];
    if (idx > 0 &&  idx < nx*ny*nz)
        atomic_add(&votes[idx], 1);

    return;
}

