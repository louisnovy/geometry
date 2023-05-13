#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

namespace py = pybind11;


void igl_bindings(py::module& m);


PYBIND11_MODULE(_geometry, m) {
    m.doc() = "C++ bindings for geometry processing";
    m.attr("__name__") = "geometry";
    m.attr("__package__") = "geometry";
    m.attr("__path__") = "geometry";
    m.attr("__file__") = "geometry/__init__.py";

    igl_bindings(m);
}
