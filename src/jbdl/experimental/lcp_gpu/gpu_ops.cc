#include "kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace gpu_lcp_jax;

namespace {

	pybind11::dict Registrations() {
		pybind11::dict dict;
		dict["gpu_lcp_double"] = EncapsulateFunction(gpu_lcp_double);
		return dict;
	}

	PYBIND11_MODULE(gpu_ops, m) {
		m.def("registrations", &Registrations);
		m.def("build_osqp_descriptor",
			[](c_int n, c_int m) {
			return PackDescriptor(OsqpDescriptor{n, m});
		});
	}
	
}  // namespace