#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

namespace py = pybind11;

void igl_bindings(py::module& m);

PYBIND11_MODULE(geometry_bindings, m) {
    igl_bindings(m);
}
