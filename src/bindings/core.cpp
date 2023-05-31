#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

namespace py = pybind11;

void bindings(py::module& m);

PYBIND11_MODULE(_geometry, m) {
    m.doc() = "C++ bindings for geometry processing";
    m.attr("__name__") = "geometry";
    m.attr("__package__") = "geometry";
    m.attr("__path__") = "geometry";
    m.attr("__file__") = "geometry/__init__.py";

    py::module igl = m.def_submodule("igl");
    igl.doc() = "libigl bindings";

    py::module fcpw = m.def_submodule("fcpw");
    fcpw.doc() = "fcpw bindings";

    bindings(m);
}



// void igl_bindings(py::module& m);

// PYBIND11_MODULE(_geometry, m) {
//     m.doc() = "C++ bindings for geometry processing";

//     py::module igl = m.def_submodule("igl");
//     igl.doc() = "libigl bindings";

//     igl_bindings(igl);
// }
