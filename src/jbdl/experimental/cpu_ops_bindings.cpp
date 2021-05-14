#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "math.hpp"
#include "qpOASES.hpp"
#include "cast_tools.hpp"

template <typename T>
pybind11::bytes PackDescriptor(const T& descriptor) {
    return pybind11::bytes(PackDescriptorAsString(descriptor));
}

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
    return pybind11::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

template <typename T>
void cpu_kepler(void *out_tuple, const void **in) {
    // Parse the inputs
    const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]);
    const T *mean_anom = reinterpret_cast<const T *>(in[1]);
    const T *ecc = reinterpret_cast<const T *>(in[2]);

    // The output is stored as a list of pointers since we have multiple outputs
    void **out = reinterpret_cast<void **>(out_tuple);
    T *sin_ecc_anom = reinterpret_cast<T *>(out[0]);
    T *cos_ecc_anom = reinterpret_cast<T *>(out[1]);

    for (std::int64_t n = 0; n < size; ++n) {
        compute_eccentric_anomaly(mean_anom[n], ecc[n], sin_ecc_anom + n, cos_ecc_anom + n);
    }
}

void lcp(const qpOASES::real_t* H,
         const qpOASES::real_t* f,
         const qpOASES::real_t* L,
         const qpOASES::real_t* k,
         const qpOASES::real_t* lb,
         const qpOASES::real_t* ub,
         const size_t nV,
         const size_t nC,
         qpOASES::real_t* primal,
         qpOASES::real_t* dual){
    int nWSR = 20;
    qpOASES::QProblem QProblemCore = qpOASES::QProblem(nV, nC, qpOASES::HST_UNKNOWN, qpOASES::BT_TRUE);
    QProblemCore.init(H, f, L, lb, ub, nullptr, k, nWSR, nullptr);
    QProblemCore.getPrimalSolution(primal);
    qpOASES::real_t qpDual[nV+nC];
    QProblemCore.getDualSolution(qpDual);
    for(int i = 0; i < nC; ++i){
        if(qpDual[i+nV] < 0.0){
            dual[i] = -1.0 * qpDual[i+nV];
        } else {
            dual[i] = 0.0;
        }
    }

    for(int i = nC; i < nC+nV; ++i){
        if(qpDual[i-nC] > 0.0){
            dual[i] = 1.0 * qpDual[i-nC];
        } else {
            dual[i] = 0.0;
        }
    }

    for(int i = nC+nV; i < nC+2*nV; ++i){
        if(qpDual[i-nC-nV] < 0.0){
            dual[i] = -1.0 * qpDual[i-nC-nV];
        } else {
            dual[i] = 0.0;
        }
    }


}

void cpu_lcp_f64(void *out_tuple, const void **in) {
    // Parse the inputs
    const std::int64_t nV = *reinterpret_cast<const std::int64_t *>(in[0]);
    const std::int64_t nC = *reinterpret_cast<const std::int64_t *>(in[1]);
    const double *H = reinterpret_cast<const double *>(in[2]);
    const double *f = reinterpret_cast<const double *>(in[3]);
    const double *L = reinterpret_cast<const double *>(in[4]);
    const double *k = reinterpret_cast<const double *>(in[5]);
    const double *lb = reinterpret_cast<const double *>(in[6]);
    const double *ub = reinterpret_cast<const double *>(in[7]);

    // The output is stored as a list of pointers since we have multiple outputs
    void **out = reinterpret_cast<void **>(out_tuple);
    double *primal = reinterpret_cast<double *>(out[0]);
    double *dual = reinterpret_cast<double *>(out[1]);
    lcp(H, f, L, k, lb, ub, nV, nC, primal, dual);
}

void cpu_lcp_f32(void *out_tuple, const void **in) {
    // Parse the inputs
    const std::int64_t nV = *reinterpret_cast<const std::int64_t *>(in[0]);
    const std::int64_t nC = *reinterpret_cast<const std::int64_t *>(in[1]);
    const float *input_H = reinterpret_cast<const float *>(in[2]);
    const float *input_f = reinterpret_cast<const float *>(in[3]);
    const float *input_L = reinterpret_cast<const float *>(in[4]);
    const float *input_k = reinterpret_cast<const float *>(in[5]);
    const float *input_lb = reinterpret_cast<const float *>(in[6]);
    const float *input_ub = reinterpret_cast<const float *>(in[7]);

    std::vector<double> H, f, L, k, lb, ub;
    for(int i=0; i<nV*nV; ++i){
        H.push_back(static_cast<double>(input_H[i]));
    }
    for(int i=0; i<nV; ++i){
        f.push_back(static_cast<double>(input_f[i]));
        lb.push_back(static_cast<double>(input_lb[i]));
        ub.push_back(static_cast<double>(input_ub[i]));
    }
    for(int i=0; i<nC * nV; ++i){
        L.push_back(static_cast<double>(input_L[i]));
    }
    for(int i=0; i<nC; ++i){
        k.push_back(static_cast<double>(input_k[i]));
    }

    std::vector<double> primal(nV, 0);
    std::vector<double> dual(nC + 2 * nV, 0);

    // The output is stored as a list of pointers since we have multiple outputs
    void **out = reinterpret_cast<void **>(out_tuple);
    float *output_primal = reinterpret_cast<float *>(out[0]);
    float *output_dual = reinterpret_cast<float *>(out[1]);
    lcp(H.data(), f.data(), L.data(), k.data(), lb.data(), ub.data(), nV, nC, primal.data(), dual.data());
    for(int i=0; i<nV; ++i){
        output_primal[i] = static_cast<float>(primal[i]);
    }
    for(int i=0; i<nC+2*nV; ++i){
        output_dual[i] = static_cast<float>(dual[i]);
    }
}



std::tuple<std::vector<qpOASES::real_t>, std::vector<qpOASES::real_t>> lcp_wrapper(std::vector<qpOASES::real_t> H,
                std::vector<qpOASES::real_t> f,
                std::vector<qpOASES::real_t> L,
                std::vector<qpOASES::real_t> k,
                std::vector<qpOASES::real_t> lb,
                std::vector<qpOASES::real_t> ub,
                size_t nV,
                size_t nC){
    std::vector<qpOASES::real_t> primal(nV, 0);
    std::vector<qpOASES::real_t> dual(nC + 2 * nV, 0);
    // std::vector<qpOASES::real_t> lcp_dual;s
    lcp(H.data(), f.data(), L.data(), k.data(), lb.data(), ub.data(), nV, nC, primal.data(), dual.data());
    // for(auto it = dual.begin() + nV; it != dual.end(); ++it){
    //     if(*it < 0){
    //         lcp_dual.push_back(-1.0 *(*it));
    //     }else{
    //         lcp_dual.push_back(0.0);           
    //     }
    // }

    // for(auto it = dual.begin(); it != dual.begin()+nV; ++it){
    //     if(*it > 0){
    //         lcp_dual.push_back(*it);
    //     }else{
    //         lcp_dual.push_back(0.0);           
    //     }
    // }

    // for(auto it = dual.begin(); it != dual.begin()+nV; ++it){
    //     if(*it < 0){
    //         lcp_dual.push_back(-1.0 *(*it));
    //     }else{
    //         lcp_dual.push_back(0.0);           
    //     }
       
    // }


    
    return std::make_tuple(primal, dual);
}

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["cpu_kepler_f32"] = EncapsulateFunction(cpu_kepler<float>);
    dict["cpu_kepler_f64"] = EncapsulateFunction(cpu_kepler<double>);
    dict["cpu_lcp_f32"] = EncapsulateFunction(cpu_lcp_f32);
    dict["cpu_lcp_f64"] = EncapsulateFunction(cpu_lcp_f64);
    return dict;
}


PYBIND11_MODULE(cpu_ops, m) {
    m.def("registrations", &Registrations);
    m.def("lcp", &lcp_wrapper);

}
