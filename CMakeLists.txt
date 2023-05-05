cmake_minimum_required(VERSION 3.11)

project(geometry DESCRIPTION "geometry bindings")

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
set(EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LIBIGL_COPYLEFT_CGAL "Use CGAL" ON)
include(libigl)


add_subdirectory(./ext/pybind11)
pybind11_add_module(
    geometry_bindings
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/igl.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/core.cpp"
)

target_link_libraries(geometry_bindings PUBLIC igl::core igl_copyleft::cgal)
target_include_directories(geometry_bindings PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/")
