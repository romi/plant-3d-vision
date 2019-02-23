#include "FMM.h"

#include <algorithm>
#include <vector>

#include "cl.h"

namespace romi {

std::string geodesic_source = {R"CLC(
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

)CLC"};

GeodesicVoting::GeodesicVoting(int nx, int ny, int nz)
    : nx(nx), ny(ny), nz(nz) {
    build_kernel(geodesic_source, "geodesic", kernel);
    init_cl_buffers();
}

void GeodesicVoting::init_cl_buffers() {
    cl::ImageFormat image_format_u(CL_INTENSITY, CL_UNORM_INT8);
    cl::ImageFormat image_format_f(CL_INTENSITY, CL_FLOAT);

    gx = cl::Image3D(context, CL_MEM_READ_ONLY, image_format_f, nx, ny, nz);
    gy = cl::Image3D(context, CL_MEM_READ_ONLY, image_format_f, nx, ny, nz);
    gz = cl::Image3D(context, CL_MEM_READ_ONLY, image_format_f, nx, ny, nz);

    values = cl::Image3D(context, CL_MEM_READ_ONLY, image_format_f, nx, ny, nz);

    votes = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * nx * ny * nz);
}

void GeodesicVoting::compute_geodesics(const float* &gx_h,
                                       const float* &gy_h,
                                       const float* &gz_h,
                                       const float* &values_h,
                                       const float* &pts_h,
                                       int n_pts,
                                       int max_iters, float step_size,
                                       std::vector<float> &votes_h) {
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
    region[0] = nz; // OK ? column first ?
    region[1] = ny;
    region[2] = nx;

    queue.enqueueWriteImage(gx, CL_TRUE, origin, region, ny, nz * ny,
                            (void *) gx_h);
    queue.enqueueWriteImage(gy, CL_TRUE, origin, region, ny, nz * ny,
                            (void *) gy_h);
    queue.enqueueWriteImage(gz, CL_TRUE, origin, region, ny, nz * ny,
                            (void *) gz_h);

    queue.enqueueWriteImage(values, CL_TRUE, origin, region, ny, nz * ny,
                            (void *) values_h);

    votes_h = std::vector<float>(nx * ny * nz);
    std::fill(votes_h.begin(), votes_h.end(), 0);

    queue.enqueueWriteBuffer(votes, CL_TRUE, 0, sizeof(float) * nx * nz * ny,
                            (void *)&votes_h.at(0));


    cl::Buffer pts(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * n_pts);
    cl::Buffer labels(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * n_pts);

    std::vector<uint8_t> labels_h(n_pts);
    std::fill(labels_h.begin(), labels_h.end(), 1);

    queue.enqueueWriteBuffer(pts, CL_TRUE, 0, sizeof(float) * n_pts,
                             (void *) pts_h);
    queue.enqueueWriteBuffer(labels, CL_TRUE, 0, sizeof(uint8_t) * n_pts,
                             (void *) &labels_h.at(0));

    uint8_t points_remain_h = true;
    cl::Buffer points_remain(context, CL_MEM_READ_WRITE, sizeof(uint8_t));

    queue.enqueueWriteBuffer(points_remain, CL_TRUE, 0, sizeof(uint8_t),
                             &points_remain_h);

    kernel.setArg(0, gx);
    kernel.setArg(1, gy);
    kernel.setArg(2, gz);
    kernel.setArg(3, values);
    kernel.setArg(4, votes);
    kernel.setArg(5, pts);
    kernel.setArg(6, labels);
    kernel.setArg(7, points_remain);
    kernel.setArg(8, step_size);

    for (size_t i = 0; i < max_iters; i++) {
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n_pts));
        queue.enqueueReadBuffer(points_remain, CL_TRUE, 0, sizeof(uint8_t),
                                &points_remain_h);
        queue.finish();
        if (!points_remain_h)
            break;
    }

    queue.enqueueReadBuffer(votes, CL_TRUE, 0, sizeof(float) * nx * nz * ny,
                           (void *)&votes_h.at(0));
}

} // namespace romi
