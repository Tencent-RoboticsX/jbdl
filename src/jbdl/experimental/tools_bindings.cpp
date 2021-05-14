#include <pybind11/pybind11.h>
#include "tools.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tools, m) {
    // optional module docstring
    m.doc() = "pybind11 example plugin";
    m.def("multiply", &multiply, "A function which adds two numbers");


    // bindings to Pet class
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &, int>())
        .def("go_for_a_walk", &Pet::go_for_a_walk)
        .def("get_hunger", &Pet::get_hunger)
        .def("get_name", &Pet::get_name);
}