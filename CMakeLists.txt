cmake_minimum_required(VERSION 3.11)

project(geometry)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(pybind11)
include(libigl)
include(fcpw)

pybind11_add_module(
    _geometry
    ./src/bindings/fcpw.cpp
    ./src/bindings/igl.cpp
    ./src/bindings/core.cpp
    )
    
target_link_libraries(
    _geometry
    PUBLIC
    # fcpw
    igl::core
    igl_copyleft::cgal
    )

target_include_directories(
    _geometry
    PUBLIC
    ./src/cpp/
    ${FCPW_EIGEN_INCLUDES}
    ${FCPW_ENOKI_INCLUDES}
    )
