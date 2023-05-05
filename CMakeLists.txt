cmake_minimum_required(VERSION 3.11)

project(geometry)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LIBIGL_COPYLEFT_CGAL "Use CGAL" ON)
include(libigl)

add_subdirectory(./ext/pybind11)
pybind11_add_module(
    geometry_bindings
    ./src/bindings/igl.cpp
    ./src/bindings/core.cpp
)

target_link_libraries(geometry_bindings PUBLIC igl::core igl_copyleft::cgal)
target_include_directories(geometry_bindings PUBLIC ./src/cpp/)
