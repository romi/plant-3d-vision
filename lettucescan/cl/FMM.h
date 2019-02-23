#ifndef FMM_h_INCLUDED
#define FMM_h_INCLUDED

#include "cl.h"

namespace romi {
class GeodesicVoting {
  public:
    GeodesicVoting(int n_x, int n_y, int n_z);
    void compute_geodesics(const float *&gx_h, const float *&gy_h,
                           const float *&gz_h, const float *&values_h,
                           const float *&pts_h, int n_pts, int max_iters,
                           float step_size, std::vector<float> &votes_h);

    void init_cl_buffers();

    cl::Kernel kernel;

    cl::Buffer votes;

    cl::Image3D gx;
    cl::Image3D gy;
    cl::Image3D gz;

    cl::Image3D values;

    int nx, ny, nz;
};
} // namespace romi

#endif // FMM_h_INCLUDED
