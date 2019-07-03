#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <cmath>
#include <cstdio>
#include <iostream>

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
    SnavelyReprojectionError(double observed_x, double observed_y, double focal,
                             double c_x, double c_y, double r_1, double r_2,
                             double l_1, double l_2)
        : observed_x(observed_x), observed_y(observed_y), focal(focal),
          c_x(c_x), c_y(c_y), r_1(r_1), r_2(r_2), l_1(l_1), l_2(l_2) {}
    template <typename T>
    bool operator()(const T *const camera, const T *const point,
                    T *residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
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
        const double &l1 = intrinsics[1];
        const double &l2 = intrinsics[2];
        T r2 = xp * xp + yp * yp;
        double distortion = 1.0 + r1 * (l1 + l2 * r2);
        // Compute final projected point position.
        const double &focal = intrinsics[0];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;
        // The error is the difference between the predicted and observed
        // position.
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;
        return true;
    }
    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y, double focal,
                                       double c_x, double c_y, double r_1,
                                       double r_2, double l_1, double l_2) {
        return (
            new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6, 3>(
                new SnavelyReprojectionError(observed_x, observed_y, focal, c_x,
                                             c_y, r_1, r_2, l_1, l_2)));
    }
    double observed_x;
    double observed_y;
    double focal;

    double c_x;
    double c_y;
    double r_1;
    double r_2;

    double l_1;
    double l_2
};

class BundleAdjustment {
  public:
    size_t num_points() { return points3d_.size() / 3; }
    void set_intrinsics(std::vector<double> camera_matrix,
                        std::vector<double> dist_coefs) {
        camera_matrix_ = camera_matrix;
        dist_coefs_ = dist_coefs;
    }
    // Add a view with given initial guess
    void add_view(std::vector<double> &extrinsics) {
        extrinsics_.insert(extrinsics_.end(), extrinsics.begin(),
                           extrinsics.end());
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

    void solve() {
        ceres::Problem problem;
        ceres::CostFunction *cost_function;
        for (int i = 0; i < num_points(); ++i) {
            cost_function = SnavelyReprojectionError::Create(
                points2d_l_[2 * i + 0], points2d_l_[2 * i + 1],
                camera_matrix_[3 * 0 + 0], camera_matrix_[3 * 0 + 2],
                camera_matrix_[3 * 1 + 2], dist_coefs_[0], dist_coefs_[1],
                dist_coefs_[2], dist_coefs_[3]);
            problem.AddResidualBlock(cost_function, NULL /* squared loss */,
                                     &extrinsics_.at(6 * view_index_l_[i]),
                                     &points3d_.at(3 * i));

            ceres::CostFunction *cost_function =
                SnavelyReprojectionError::Create(points2d_r_[2 * i + 0],
                                                 points2d_r_[2 * i + 1],
                                                 &intrinsics_.at(0));
            problem.AddResidualBlock(cost_function, NULL /* squared loss */,
                                     &extrinsics_.at(6 * view_index_r_[i]),
                                     &points3d_.at(3 * i));
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";
    }

  private:
    std::vector<double> points2d_l_; // Observations
    std::vector<double> points2d_r_; // Observations

    std::vector<double> points3d_;

    std::vector<double> extrinsics_; // Camera poses

    std::vector<double> camera_matrix_; // Camera parameters
    std::vector<double> dist_coefs_;    // Camera parameters

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
        .def("solve", &BundleAdjustment::solve);
}
