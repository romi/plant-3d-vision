#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FMM.h"
#include "SpaceCarving.h"
#include "cl.h"

namespace py = pybind11;
using namespace romi;

void process_view_python(
    SpaceCarving &instance, const Vec4f &intrinsics, const Mat3f &rot,
    const Vec3f &tvec,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>
        &pymask) {
    const uint8_t *mask = pymask.data();
    instance.process_view(intrinsics, rot, tvec, mask);
}

py::array_t<uint8_t> labels_python(SpaceCarving &instance) {
    py::array_t<uint8_t> labels(
        py::buffer_info(&instance.labels.at(0), sizeof(uint8_t),
                        py::format_descriptor<uint8_t>::value, 1,
                        {instance.labels.size()}, {sizeof(uint8_t)}));
    return labels;
}

py::array_t<float> centers_python(SpaceCarving &instance) {
    size_t size[2] = {instance.centers.size() / 3, 3};
    size_t strides[2] = {sizeof(float) * 3, sizeof(float)};
    py::array_t<float> centers(
        py::buffer_info(&instance.centers.at(0), sizeof(float),
                        py::format_descriptor<float>::value, 2, size, strides));
    return centers;
}

py::array_t<float> geodesic_voting_python(
    GeodesicVoting &instance,
    const py::array_t<float, py::array::c_style | py::array::forcecast> gx,
    const py::array_t<float, py::array::c_style | py::array::forcecast> gy,
    const py::array_t<float, py::array::c_style | py::array::forcecast> gz,
    const py::array_t<float, py::array::c_style | py::array::forcecast> values,
    const py::array_t<float, py::array::c_style | py::array::forcecast> pts,
    int max_iters, float step_size) {

    assert(pts.ndim() == 2);
    assert(pts.shape(1) == 3);

    assert(gx.ndim() == 3);
    assert(gy.ndim() == 3);
    assert(gz.ndim() == 3);
    assert(values.ndim() == 3);

    int n_pts = pts.shape(0);

    const float *gx_h = gx.data();
    const float *gy_h = gy.data();
    const float *gz_h = gz.data();
    const float *values_h = values.data();
    const float *pts_h = pts.data();

    size_t nx = instance.nx;
    size_t ny = instance.ny;
    size_t nz = instance.nz;

    std::vector<float> votes_h(nx * ny * nz);
    instance.compute_geodesics(gx_h, gy_h, gz_h, values_h, pts_h, n_pts,
                               max_iters, step_size, votes_h);

    size_t strides[3] = {nz * ny * sizeof(float),
                         ny * sizeof(float), sizeof(float)};

    size_t size[3] = {nx, ny, nz};

    py::array_t<float> votes(
        py::buffer_info(&votes_h.at(0), sizeof(float),
                        py::format_descriptor<float>::value, 3, size, strides));
    return votes;
}

PYBIND11_MODULE(cl, m) {
    py::class_<SpaceCarving>(m, "SpaceCarving")
        .def(py::init<const Vec3f &, const Vec3f &, float, size_t, size_t>())
        .def("reset_labels", &SpaceCarving::reset_labels)
        .def("process_view", process_view_python)
        .def("labels", labels_python)
        .def("centers", centers_python);
    py::class_<GeodesicVoting>(m, "GeodesicVoting")
        .def(py::init<float, float, float>())
        .def("run", geodesic_voting_python);
    // Add bindings here
    m.def("init_opencl", init_opencl);
}
