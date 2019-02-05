# Created by the script cgal_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.

project( src )


cmake_minimum_required(VERSION 2.8.11)

find_package(pybind11 REQUIRED)
find_package(OpenCL REQUIRED)

pybind11_add_module(spacecarving SHARED bindings.cc SpaceCarving.cc)
target_link_libraries(spacecarving ${OpenCL_LIBRARY})
target_include_directories(spacecarving PUBLIC ${OpenCL_INCLUDE_DIRS})

