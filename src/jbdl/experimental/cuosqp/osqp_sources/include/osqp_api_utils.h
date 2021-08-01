#ifndef OSQP_API_UTILS_H
#define OSQP_API_UTILS_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

/* Types required by the OSQP utility functions */
# include "osqp_configure.h"
# include "osqp_api_types.h"

/********************
* OSQP User utilities  *
********************/
/**
 * Populates a Compressed-Column-Sparse matrix from existing arrays
    (just assigns the pointers - no malloc or copying is done)
 * @param  m     First dimension
 * @param  n     Second dimension
 * @param  nzmax Maximum number of nonzero elements
 * @param  x     Vector of data
 * @param  i     Vector of row indices
 * @param  p     Vector of column pointers
 * @return       New matrix pointer
 */
void csc_set_data(csc* M,
                  c_int    m,
                  c_int    n,
                  c_int    nzmax,
                  c_float *x,
                  c_int   *i,
                  c_int   *p);

# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef OSQP_API_FUNCTIONS_H
