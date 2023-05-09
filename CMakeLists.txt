# cmake_minimum_required(VERSION 3.11)

# project(geometry)

# set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# option(LIBIGL_COPYLEFT_CGAL "Use CGAL" ON)
# include(libigl)

# add_subdirectory(./ext/pybind11)
# pybind11_add_module(
#     geometry_bindings
#     ./src/bindings/igl.cpp
#     ./src/bindings/core.cpp
# )

# target_link_libraries(geometry_bindings PUBLIC igl::core igl_copyleft::cgal)
# target_include_directories(geometry_bindings PUBLIC ./src/cpp/)



cmake_minimum_required(VERSION 3.15)

project(geometry)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Get NumpyEigen if we haven't already
set(EXTERNAL_DEP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
include(DownloadExternalDeps)
download_dep(numpyeigen
  GIT_REPOSITORY https://github.com/fwilliams/numpyeigen.git
  GIT_TAG 4916d926aa2b939bd8f625c7537563a1575dafe9
)
list(APPEND CMAKE_MODULE_PATH ${EXTERNAL_DEP_DIR}/numpyeigen/cmake)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(numpyeigen) # This will define Eigen3::Eigen if we enabled NPE_WITH_EIGEN

# Get libigl if we haven't already
option(LIBIGL_COPYLEFT_CGAL "Use CGAL" ON)
include(libigl)

file(GLOB IGL_SOURCES src/bindings/igl/*.cpp)
npe_add_module(_geometry
  BINDING_SOURCES
  ${IGL_SOURCES}
  )

set_target_properties(_geometry PROPERTIES COMPILE_FLAGS "-fvisibility=hidden -msse3")

# enable_testing()
# add_subdirectory(tests)
