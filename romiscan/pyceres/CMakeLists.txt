# Created by the script ceres_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.
cmake_minimum_required(VERSION 2.8.11)

# CGAL and its components
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_library(pyceres SHARED ceres.cc)
set_target_properties(pyceres PROPERTIES PREFIX "")
target_link_libraries(pyceres ${CERES_LIBRARIES})
target_include_directories(pyceres PUBLIC ${CERES_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR})

