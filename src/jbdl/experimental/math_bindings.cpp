#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "math.hpp"
#include <map>

namespace py = pybind11;


template <typename T>
std::map<std::string, T> compute_eccentric_anomaly_wrapper(T mean_anom, T ecc){
    T sin_ecc_anom;
    T cos_ecc_anom;
    compute_eccentric_anomaly(mean_anom, ecc, &sin_ecc_anom, &cos_ecc_anom);
    std::map<std::string, T>  ecc_anom{
        {"sin_ecc_anom", sin_ecc_anom}, {"cos_ecc_anom", cos_ecc_anom}};
    
    return ecc_anom;

} 

template std::map<std::string, float> compute_eccentric_anomaly_wrapper<float>(float mean_anom, float ecc);
template std::map<std::string, double> compute_eccentric_anomaly_wrapper<double>(double mean_anom, double ecc);


PYBIND11_MODULE(math, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");

    m.def("subtract", &subtract, "A function which subtracts two numbers");

    m.def("compute_eccentric_anomaly", &compute_eccentric_anomaly_wrapper<double>);
}