#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "SpaceCarving.h"

namespace py = pybind11;
using namespace space_carving;

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

PYBIND11_MODULE(space_carving, m) {
    py::class_<SpaceCarving>(m, "SpaceCarving")
        .def(py::init<const Vec3f &, const Vec3f &, float, size_t, size_t>())
        .def("reset_labels", &SpaceCarving::reset_labels)
        .def("process_view", process_view_python)
        .def("labels", labels_python)
        .def("centers", centers_python);
    // Add bindings here
    m.def("init_opencl", init_opencl);
}
