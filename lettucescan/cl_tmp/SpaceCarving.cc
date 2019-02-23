#include "SpaceCarving.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "cl.h"

namespace romi {

std::string kernel_check_source = {R"CLC(
__kernel void check(__read_only image2d_t mask,
            __global uchar* labels,
            __global float* centers,
            __global float* intrinsics,
            __global float* rot,
            __global float* tvec) {
const sampler_t samplerA = CLK_NORMALIZED_COORDS_FALSE |
                       CLK_ADDRESS_NONE |
                       CLK_FILTER_NEAREST;
int i = get_global_id(0);
if (labels[i] == 1)
{
return;
}

float f_x = intrinsics[0];
float f_y = intrinsics[1];
float c_x = intrinsics[2];
float c_y = intrinsics[3];

float p_x = rot[0] * centers[3*i + 0] +
        rot[1] * centers[3*i + 1] +
        rot[2] * centers[3*i + 2] + tvec[0];

float p_y = rot[3] * centers[3*i + 0] +
        rot[4] * centers[3*i + 1] +
        rot[5] * centers[3*i + 2] + tvec[1];

float p_z = rot[6] * centers[3*i + 0] +
        rot[7] * centers[3*i + 1] +
        rot[8] * centers[3*i + 2] + tvec[2];

if (p_z < 0) {
return;
}
p_x = p_x/p_z * f_x + c_x;
p_y = p_y/p_z * f_y + c_y;

int p_x_int = (int) p_x;
int p_y_int = (int) p_y;
int2 image_coord = {p_x_int, p_y_int};

if (p_x_int < 0 || p_x_int > get_image_width(mask)-1) {
return;
}
if (p_y_int < 0 || p_y_int > get_image_height(mask)-1) {
return;
}
if (read_imageui(mask, samplerA, image_coord).x == 0) {
    labels[i] = 1;
} else if (labels[i] == 0) { // Mark a voxel the first time it is seen
    labels[i] = 2;
}
}
)CLC"};

SpaceCarving::SpaceCarving(const Vec3f &center, const Vec3f &width,
                           float voxel_size, size_t mask_width,
                           size_t mask_height)
    : width(mask_width), height(mask_height), voxel_size(voxel_size) {

    build_kernel(kernel_check_source, "check", kernel);

    int half_n_subdiv_x = (int)(0.5f * width[0] / voxel_size);
    int half_n_subdiv_y = (int)(0.5f * width[1] / voxel_size);
    int half_n_subdiv_z = (int)(0.5f * width[2] / voxel_size);

    n_voxels = (2 * half_n_subdiv_x + 1) * (2 * half_n_subdiv_y + 1) *
               (2 * half_n_subdiv_z + 1);

    centers = std::vector<float>(3 * n_voxels);
    size_t l = 0;
    for (int i = -half_n_subdiv_x; i <= half_n_subdiv_x; i++) {
        for (int j = -half_n_subdiv_y; j <= half_n_subdiv_y; j++) {
            for (int k = -half_n_subdiv_z; k <= half_n_subdiv_z; k++) {
                centers[l] = center[0] + i * voxel_size;
                centers[l + 1] = center[1] + j * voxel_size;
                centers[l + 2] = center[2] + k * voxel_size;
                l += 3;
            }
        }
    }

    labels = std::vector<uint8_t>(n_voxels);
    std::fill(labels.begin(), labels.end(), 0);
    init_cl_buffers();
}

void SpaceCarving::init_cl_buffers() {
    centers_device =
        cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * 3 * n_voxels);
    labels_device =
        cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(uint8_t) * n_voxels);

    intrinsics_device =
        cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * 4);
    rot_device = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * 9);
    tvec_device = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * 3);

    cl::ImageFormat image_format(CL_INTENSITY, CL_UNORM_INT8);
    mask_device =
        cl::Image2D(context, CL_MEM_READ_ONLY, image_format, width, height);

    queue.enqueueWriteBuffer(centers_device, CL_TRUE, 0,
                             n_voxels * sizeof(float) * 3, &centers.at(0));
    queue.enqueueWriteBuffer(labels_device, CL_TRUE, 0,
                             n_voxels * sizeof(uint8_t), &labels.at(0));
    queue.finish();
}

void SpaceCarving::sync_to_device() {
    queue.enqueueWriteBuffer(labels_device, CL_TRUE, 0,
                             n_voxels * sizeof(uint8_t), &labels.at(0));
    queue.finish();
}

void SpaceCarving::sync_to_host() {
    queue.enqueueReadBuffer(labels_device, CL_TRUE, 0,
                            n_voxels * sizeof(uint8_t), &labels.at(0));
    queue.finish();
}

void SpaceCarving::reset_labels() {
    std::fill(labels.begin(), labels.end(), 0);
    sync_to_device();
}

void SpaceCarving::process_view(const Vec4f &intrinsics, const Mat3f &rot,
                                const Vec3f &tvec, const uint8_t *mask_pixels) {
    try {
        queue.enqueueWriteBuffer(intrinsics_device, CL_TRUE, 0,
                                 4 * sizeof(float), &intrinsics.at(0));
        queue.enqueueWriteBuffer(rot_device, CL_TRUE, 0, 9 * sizeof(float),
                                 &rot.at(0));
        queue.enqueueWriteBuffer(tvec_device, CL_TRUE, 0, 3 * sizeof(float),
                                 &tvec.at(0));
        cl::size_t<3> origin;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        cl::size_t<3> region;
        region[0] = width;
        region[1] = height;
        region[2] = 1;

        queue.enqueueWriteImage(mask_device, CL_TRUE, origin, region, width, 0,
                                (void *)&mask_pixels[0]);

        kernel.setArg(0, mask_device);
        kernel.setArg(1, labels_device);
        kernel.setArg(2, centers_device);
        kernel.setArg(3, intrinsics_device);
        kernel.setArg(4, rot_device);
        kernel.setArg(5, tvec_device);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(n_voxels));
    } catch (cl::Error err) {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
        throw err;
    }
    sync_to_host();
}

} // namespace romi
