#include <Eigen/Dense>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>

namespace py = pybind11;

void hough_transform(py::array_t<double> volume, py::array_t<double> output,
                     std::vector<double> a_x_values,
                     std::vector<double> b_x_values,
                     std::vector<double> a_y_values,
                     std::vector<double> b_y_values) {
    assert(volume.ndim() == 3);

    auto n_x = volume.shape()[0];
    auto n_y = volume.shape()[1];
    auto n_z = volume.shape()[2];

    auto n_u = a_x_values.size();
    auto n_v = b_x_values.size();
    auto n_w = a_y_values.size();
    auto n_t = b_y_values.size();

    assert(output.shape()[0] == n_u);
    assert(output.shape()[1] == n_v);
    assert(output.shape()[2] == n_w);
    assert(output.shape()[3] == n_t);

    size_t i = 0;
    auto r = output.mutable_unchecked<4>(); 
    auto vol = volume.unchecked<3>();
    for (int u = 0; u < n_u; u++) {
        double a_x = a_x_values[u];
        std::cout << "a_x = " << a_x << std::endl;
        for (int v = 0; v < n_v; v++) {
            double b_x = b_x_values[v];
            for (int w = 0; w < n_w; w++) {
                double a_y = a_y_values[w];
                for (int t = 0; t < n_t; t++) {
                    double b_y = b_y_values[t];
                    for (int z = 0; z < n_z; z++) {
                        int x = llround(a_x * z + b_x);
                        int y = llround(a_y * z + b_y);
                        if (x > 0 && x < n_x && y > 0 && y < n_y) {
                            r(u, v, w, t) += vol(x, y, z);
                        }
                    }
                    i++;
                }
            }
        }
    }
}

PYBIND11_MODULE(hough, m) {
    // Add bindings here 
    m.def("hough_transform", hough_transform);
}
