# Created by the script cgal_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.
cmake_minimum_required(VERSION 2.8.11)

find_package(pybind11 REQUIRED)

# include for local directory

# include for local package

find_package(Eigen3 REQUIRED)

set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3/")

include_directories("/usr/include/eigen3")

pybind11_add_module(hough SHARED hough.cc)
