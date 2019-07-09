#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <cassert>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y, double* camera_model)
        : observed_x(observed_x), observed_y(observed_y),
          camera_model(camera_model) {}
    template <typename T>
    bool operator()(const T *const camera, const T *const point,
                    T *residuals) const {

        double focal = camera_model[0];
        double cx = camera_model[1];
        double cy = camera_model[2];

        double k1 = camera_model[3];
        double k2 = camera_model[4];

        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];
        // Apply second and fourth order radial distortion.
        T r2 = xp * xp + yp * yp;
        T distortion = 1.0 + r2 * (k1 + k2 * r2);
        // Compute final projected point position.
        T predicted_x = focal * distortion * xp + cx;
        T predicted_y = focal * distortion * yp + cy;
        // The error is the difference between the predicted and observed
        // position.
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;
        return true;
    }
    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y,
                                       double* camera_model) {
        return (
            new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6, 3>(
                new SnavelyReprojectionError(observed_x, observed_y, camera_model)));
    }
    double observed_x;
    double observed_y;
    double* camera_model;
};

class BundleAdjustment {
  public:
    size_t num_points() { return points3d_.size() / 3; }
    size_t num_views() { return extrinsics_.size() / 6; }
    void set_intrinsics(std::vector<double> camera_model) {
        camera_model_ = camera_model;
    }
    // Add a view with given initial guess
    size_t add_view(std::vector<double> &extrinsics) {
        assert(extrinsics.size() == 6);
        extrinsics_.insert(extrinsics_.end(), extrinsics.begin(),
                           extrinsics.end());
        return (extrinsics_.size() - 1) / 6;
    }

    // Add a pair with given observations and initial guess for the 3d points
    void add_pair(size_t i_l, size_t i_r, std::vector<double> &points2d_l,
                  std::vector<double> &points2d_r,
                  std::vector<double> &points3d) {
        points2d_l_.insert(points2d_l_.end(), points2d_l.begin(),
                           points2d_l.end());
        points2d_r_.insert(points2d_r_.end(), points2d_r.begin(),
                           points2d_r.end());
        points3d_.insert(points3d_.end(), points3d.begin(), points3d.end());

        view_index_l_.insert(view_index_l_.end(), points2d_l_.size(), i_l);
        view_index_r_.insert(view_index_r_.end(), points2d_r_.size(), i_r);
    }

    std::vector<std::pair<std::vector<double>, std::vector<double>>> get_extrinsics() {
        std::vector<std::pair<std::vector<double>, std::vector<double>>> res;
        for (int i = 0; i < num_views(); ++i) {
            std::vector<double> rodrigues;
            for (int j = 0; j < 3; j++) {
                rodrigues.push_back(extrinsics_[6*i + j]);
            }
            std::vector<double> tvec;
            for (int j = 0; j < 3; j++) {
                tvec.push_back(extrinsics_[6*i + 3 + j]);
            }
            res.push_back({rodrigues, tvec});
        }
        return res;
    }

    void init_cost_function() {
          for (int i = 0; i < num_points(); ++i) {
            ceres::CostFunction *cost_function_l = SnavelyReprojectionError::Create(
                points2d_l_[2 * i + 0], points2d_l_[2 * i + 1],
                &camera_model_.at(0));
            problem.AddResidualBlock(cost_function_l, NULL /* squared loss */,
                                     &extrinsics_.at(6 * view_index_l_[i]),
                                     &points3d_.at(3 * i));

            ceres::CostFunction *cost_function_r =
                SnavelyReprojectionError::Create(points2d_r_[2 * i + 0],
                                                 points2d_r_[2 * i + 1],
                                                 &camera_model_.at(0));
            problem.AddResidualBlock(cost_function_r, NULL /* squared loss */,
                                     &extrinsics_.at(6 * view_index_r_[i]),
                                     &points3d_.at(3 * i));
        }
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true; 
        }


    void solve() {
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";
    }

  private:
    ceres::Solver::Options options;
    ceres::Problem problem;
    ceres::Solver::Summary summary;

    std::vector<double> points2d_l_; // Observations
    std::vector<double> points2d_r_; // Observations

    std::vector<double> points3d_;

    std::vector<double> extrinsics_; // Camera poses

    std::vector<double> camera_model_; // Camera parameters

    std::vector<size_t> view_index_l_;
    std::vector<size_t> view_index_r_;
};

PYBIND11_MODULE(pyceres, m) {
    // Add bindings here
    py::class_<BundleAdjustment>(m, "BundleAdjustment")
        .def(py::init())
        .def("set_intrinsics", &BundleAdjustment::set_intrinsics)
        .def("add_view", &BundleAdjustment::add_view)
        .def("add_pair", &BundleAdjustment::add_pair)
        .def("init_cost_function", &BundleAdjustment::init_cost_function)
        .def("get_extrinsics", &BundleAdjustment::get_extrinsics)
        .def("solve", &BundleAdjustment::solve);
}
