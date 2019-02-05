# Created by the script cgal_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.

project( src )


cmake_minimum_required(VERSION 2.8.11)

find_package(pybind11 REQUIRED)
find_package(OpenCL REQUIRED)

pybind11_add_module(space_carving SHARED bindings.cc SpaceCarving.cc)
target_link_libraries(space_carving ${OpenCL_LIBRARY})
target_include_directories(space_carving PUBLIC ${OpenCL_INCLUDE_DIRS})

