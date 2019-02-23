#ifndef SPACE_CARVING_H
#define SPACE_CARVING_H

#include <array>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.cpp>
#else
#include <CL/cl.hpp>
#endif

using Vec3f = std::array<float, 3>;
using Vec4f = std::array<float, 4>;
using Vec3s = std::array<size_t, 3>;
using Mat3f = std::array<float, 9>;

namespace romi {

void init_opencl(int platform_id, int device_id);
void build_kernels();

class SpaceCarving {
    float voxel_size;
    size_t n_voxels;
    size_t width;
    size_t height;

    cl::Kernel kernel;

    cl::Buffer centers_device;
    cl::Buffer labels_device;

    cl::Buffer intrinsics_device;
    cl::Buffer rot_device;
    cl::Buffer tvec_device;

    cl::Image2D mask_device;

    void sync_to_host();
    void sync_to_device();
    void init_cl_buffers();

  public:
    SpaceCarving(const Vec3f &center, const Vec3f &width, float voxel_size,
                 size_t mask_width, size_t mask_height);
    void reset_labels();
    void process_view(const Vec4f &intrinsics, const Mat3f &rot,
                      const Vec3f &tvec, const uint8_t *mask);

    std::vector<float> centers;
    std::vector<uint8_t> labels;
};
} // namespace space_carving

#endif
