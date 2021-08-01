/**
 *  Copyright (c) 2019-2021 ETH Zurich, Automatic Control Lab,
 *  Michel Schubiger, Goran Banjac.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef CUDA_PCG_H
# define CUDA_PCG_H
 
#include "cuda_pcg_interface.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 *  Preconditioned Conjugate Gradient (PCG) algorithm.
 *  Computes an approximate solution to the linear system
 *
 *       K * x = rhs
 *
 *  The solution is stored in s->d_x.
 *  The function returns the number of PCG iterations evaluated.
 */
c_int cuda_pcg_alg(cudapcg_solver *s,
                   c_float         eps,
                   c_int           max_iter);

void cuda_pcg_update_precond(cudapcg_solver *s,
                             c_int           P_updated,
                             c_int           A_updated,
                             c_int           R_updated);


#ifdef __cplusplus
}
#endif

#endif /* ifndef CUDA_PCG_H */
