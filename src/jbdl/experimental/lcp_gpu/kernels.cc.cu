#include "kernel_helpers.h"
#include "kernels.h"
#include <osqp.h>
#include <cuda.h>
#include <cuda_runtime.h> 

namespace gpu_lcp_jax {

namespace {


template <typename T>
__global__ void P_processing(const T *P, c_int *P_p, const c_int n){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    for (int k = 0; k < n; k++) {
        if (P[k*n+i] != 0) {
              P_p[i+1] += 1;
        }
	}

}

template <typename T>
__global__ void A_processing(const T *A, c_int *A_p, const c_int n, const c_int m){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    for (int k = 0; k < m; k++) {
        if (A[k*n+i] != 0) {
              A_p[i+1] += 1;
        }
	}
}


template <typename T>
__global__ void get_the_result(const T *x, const T *y, T *primal, T *dual, const c_int m){
        int idx = blockIdx.x;
        int idy = blockIdx.y;

        primal[idx] = x[idx];
        if (idy < m) {
              dual[idy] = y[idy];
        }
        else{
              dual[idy] = 0.0;
        }
}

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}


template <typename T>
void gpu_lcp_template(cudaStream_t stream, void **buffers,
    const char *opaque, std::size_t opaque_len){

    const OsqpDescriptor &d = *UnpackDescriptor<OsqpDescriptor>(opaque, opaque_len);
    const c_int nn = d.nn;
    const c_int nm = d.nm;

    const T *P = reinterpret_cast<const T *>(buffers[0]);
    const T *q = reinterpret_cast<const T *>(buffers[1]);
    const T *A = reinterpret_cast<const T *>(buffers[2]);
    const T *l = reinterpret_cast<const T *>(buffers[3]);
    const T *u = reinterpret_cast<const T *>(buffers[4]);
    T *primal = reinterpret_cast<T *>(buffers[5]);
    T *dual = reinterpret_cast<T *>(buffers[6]);

    /* allocate indptr for matrix P on gpu */
	c_int P_p[nn+1] = {0};
	c_int *P_p_cuda;
	cudaMalloc((void**) &P_p_cuda, (nn + 1) * sizeof(c_int));
	cudaMemcpy(P_p_cuda, P_p, (nn + 1) * sizeof(c_int), cudaMemcpyHostToDevice);

    dim3 dimGrid_P(1);
    dim3 dimBlock_P(nn);
	P_processing<T><<<dimGrid_P, dimBlock_P, 0, stream>>>(P, (c_int *)P_p_cuda, nn);

    ThrowIfError(cudaGetLastError());

    /* get P's indptr */
	cudaMemcpy(P_p, P_p_cuda, (nn + 1) * sizeof(c_int), cudaMemcpyDeviceToHost);

    /* calculate P's csc form */
    T *P_malloc;
	P_malloc = (T*)malloc((nn*nn) * sizeof(T));
    cudaMemcpy(P_malloc, P, nn * nn * sizeof(T), cudaMemcpyDeviceToHost);

    c_int NNZ_P = 0;
    for (int k = 0; k < (nn+1); k++) {
        NNZ_P += P_p[k];
	}

    T *P_x;
	c_int *P_i;
	P_x = (T*)malloc(NNZ_P * sizeof(T));
	P_i = (c_int*)malloc(NNZ_P * sizeof(c_int));

    int count_i = 0;
	for (int col = 0; col < nn; col++) {
	    for (int row = 0; row < nn; row++) {
	         if (P_malloc[nn*row + col] != 0) {
	               P_x[count_i] = P_malloc[nn*row + col];
	               P_i[count_i] = row;
	               count_i++;
	         }
	    }
	    if (col > 0) {
			P_p[col] += P_p[col-1];
		}
	}
	P_p[nn] = count_i;
	cudaFree(P_p_cuda);
	free(P_malloc);

	/* allocate indptr for matrix A on gpu */
	c_int A_p[nn+1] = {0};
	c_int *A_p_cuda;
	cudaMalloc((void**) &A_p_cuda, (nn + 1) * sizeof(c_int));
	cudaMemcpy(A_p_cuda, A_p, (nn + 1) * sizeof(c_int), cudaMemcpyHostToDevice);

    dim3 dimGrid_A(1);
    dim3 dimBlock_A(nn);
    A_processing<T><<<dimGrid_A, dimBlock_A, 0, stream>>>(A, (c_int *)A_p_cuda, nn, nm);

    ThrowIfError(cudaGetLastError());

    /* get A's indptr */
	cudaMemcpy(A_p, A_p_cuda, (nn + 1) * sizeof(c_int), cudaMemcpyDeviceToHost);

    /* calculate A's csc form */
    T *A_malloc;
	A_malloc = (T*)malloc((nn*nm) * sizeof(T));
    cudaMemcpy(A_malloc, A, nn * nm * sizeof(T), cudaMemcpyDeviceToHost);

	c_int NNZ_A = 0;
    for (int k = 0; k < (nn+1); k++) {
        NNZ_A += A_p[k];
	}

    T *A_x;
	c_int *A_i;
	A_x = (T*)malloc(NNZ_A * sizeof(T));
	A_i = (c_int*)malloc(NNZ_A * sizeof(c_int));

    count_i = 0;
	for (int col = 0; col < nn; col++) {
	    for (int row = 0; row < nm; row++) {
	         if (A_malloc[nn*row + col] != 0) {
	               A_x[count_i] = A_malloc[nn*row + col];
	               A_i[count_i] = row;
	               count_i++;
	         }
	    }
	    if (col > 0) {
			A_p[col] += A_p[col-1];
		}
	}
	A_p[nn] = count_i;
	cudaFree(A_p_cuda);
	free(A_malloc);

     /* prepare other data for cpu */
    T q_malloc[nn];
	T l_malloc[nm];
	T u_malloc[nm];
	cudaMemcpy(q_malloc, q, nn * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(l_malloc, l, nm * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(u_malloc, u, nm * sizeof(T), cudaMemcpyDeviceToHost);


    /* solve the QP problem */
	c_int exitflag = 0;
	OSQPSolver   *solver;
	OSQPSettings *settings;
	csc *P_csc = (csc *)malloc(sizeof(csc));
	csc *A_csc = (csc *)malloc(sizeof(csc));

	/* Populate matrices */
	csc_set_data(A_csc, nm, nn, NNZ_A, A_x, A_i, A_p);
	csc_set_data(P_csc, nn, nn, NNZ_P, P_x, P_i, P_p);

	settings = (OSQPSettings *)malloc(sizeof(OSQPSettings));
	if (settings) osqp_set_default_settings(settings);
	settings->polish = 0;
	settings->verbose = 0;

	/* Setup workspace */
	exitflag = osqp_setup(&solver, P_csc, q_malloc, A_csc, l_malloc, u_malloc, nm, nn, settings);

	/* Solve Problem */
	osqp_solve(solver);

    /* send the solutions to gpu */
	T *solution_x_cuda;
	cudaMalloc((void**) &solution_x_cuda, nn * sizeof(T));
    cudaMemcpy(solution_x_cuda, solver->solution->x, nn * sizeof(T), cudaMemcpyHostToDevice);

	T *solution_y_cuda;
	cudaMalloc((void**) &solution_y_cuda, nm * sizeof(T));
    cudaMemcpy(solution_y_cuda, solver->solution->y, nm * sizeof(T), cudaMemcpyHostToDevice);

    /* get the result from gpu */
    dim3 dimGrid(nn, nm+nn);
    dim3 dimBlock(1);
	get_the_result<T><<<dimGrid, dimBlock, 0, stream>>>(solution_x_cuda, solution_y_cuda, primal, dual, nm);

	ThrowIfError(cudaGetLastError());

    cudaFree(solution_x_cuda);
    cudaFree(solution_y_cuda);
}
}  // namespace


void gpu_lcp_double(cudaStream_t stream, void **buffers, const char *opaque,
                    std::size_t opaque_len) {
  gpu_lcp_template<c_float>(stream, buffers, opaque, opaque_len);
}

}  // namespace gpu_lcp_jax