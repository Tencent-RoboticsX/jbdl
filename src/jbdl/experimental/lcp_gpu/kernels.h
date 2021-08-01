#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <osqp.h>

namespace gpu_lcp_jax {

	struct OsqpDescriptor{
		c_int nn;
		c_int nm;
	};

	void gpu_lcp_double(cudaStream_t stream, void **buffers, const char *opaque,
		std::size_t opaque_len);


}  // gpu_lcp_jax
