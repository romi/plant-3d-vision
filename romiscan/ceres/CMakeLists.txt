# Created by the script ceres_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.
cmake_minimum_required(VERSION 2.8.11)

# CGAL and its components
find_package( CGAL QUIET COMPONENTS  )

if ( NOT CGAL_FOUND )

  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()  

endif()

# include helper file
include( ${CGAL_USE_FILE} )


# Boost and its components
find_package( Boost REQUIRED )

if ( NOT Boost_FOUND )

  message(STATUS "This project requires the Boost library, and will not be compiled.")

  return()  

endif()

# find_package(pybind11 REQUIRED)

# include for local directory

# include for local package

find_package(Eigen3 REQUIRED)

set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3/")

include_directories("/usr/include/eigen3")

add_definitions(-DCGAL_EIGEN3_ENABLED)


# create_single_source_ceres_program( "python_ceres_interface.cc" )
#pybind11_add_module(ceres SHARED ceres.cc)

add_library(ceres SHARED ceres.cc)
set_target_properties(ceres PROPERTIES PREFIX "")
target_link_libraries(ceres ${CGAL_LIBRARIES})
target_include_directories(ceres PUBLIC ${CGAL_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR})

