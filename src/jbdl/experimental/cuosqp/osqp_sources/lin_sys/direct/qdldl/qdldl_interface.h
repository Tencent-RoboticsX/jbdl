#ifndef QDLDL_INTERFACE_H
#define QDLDL_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "osqp.h"
#include "types.h"  //OSQPMatrix and OSQPVector[fi] types
#include "qdldl_types.h"

/**
 * QDLDL solver structure
 */
typedef struct qdldl qdldl_solver;

struct qdldl {
    enum linsys_solver_type type;

    /**
     * @name Functions
     * @{
     */
    c_int (*solve)(struct qdldl *self,
                   OSQPVectorf  *b,
                   c_int         admm_iter);

    void (*warm_start)(struct qdldl      *self,
                       const OSQPVectorf *x);

#ifndef EMBEDDED
    void (*free)(struct qdldl *self); ///< Free workspace (only if desktop)
#endif

    // This used only in non embedded or embedded 2 version
#if EMBEDDED != 1
    c_int (*update_matrices)(struct qdldl     *self,
                             const OSQPMatrix *P,
                             const OSQPMatrix *A);         ///< Update solver matrices

    c_int (*update_rho_vec)(struct qdldl      *self,
                            const OSQPVectorf *rho_vec,
                            c_float            rho_sc);    ///< Update rho_vec parameter
#endif

#ifndef EMBEDDED
    c_int nthreads;
#endif
    /** @} */

    /**
     * @name Attributes
     * @{
     */
    csc *L;                 ///< lower triangular matrix in LDL factorization
    c_float *Dinv;          ///< inverse of diag matrix in LDL (as a vector)
    c_int   *P;             ///< permutation of KKT matrix for factorization
    c_float *bp;            ///< workspace memory for solves
    c_float *sol;           ///< solution to the KKT system
    c_float *rho_inv_vec;   ///< parameter vector
    c_float sigma;          ///< scalar parameter
    c_float rho_inv;        ///< scalar parameter (used if rho_inv_vec == NULL)
#ifndef EMBEDDED
    c_int polish;           ///< polishing flag
#endif
    c_int n;                ///< number of QP variables
    c_int m;                ///< number of QP constraints


#if EMBEDDED != 1
    // These are required for matrix updates
    c_int * Pdiag_idx, Pdiag_n;  ///< index and number of diagonal elements in P
    csc   * KKT;                 ///< Permuted KKT matrix in sparse form (used to update P and A matrices)
    c_int * PtoKKT, * AtoKKT;    ///< Index of elements from P and A to KKT matrix
    c_int * rhotoKKT;            ///< Index of rho places in KKT matrix
    // QDLDL Numeric workspace
    QDLDL_float *D;
    QDLDL_int   *etree;
    QDLDL_int   *Lnz;
    QDLDL_int   *iwork;
    QDLDL_bool  *bwork;
    QDLDL_float *fwork;
#endif

    /** @} */
};



/**
 * Initialize QDLDL Solver
 *
 * @param  s         Pointer to a private structure
 * @param  P         Cost function matrix (upper triangular form)
 * @param  A         Constraints matrix
 * @param  rho_vec   Algorithm parameter. If polish, then rho_vec = OSQP_NULL.
 * @param  settings  Solver settings
 * @param  polish    Flag whether we are initializing for polish or not
 * @return           Exitflag for error (0 if no errors)
 */
c_int init_linsys_solver_qdldl(qdldl_solver      **sp,
                               const OSQPMatrix   *P,
                               const OSQPMatrix   *A,
                               const OSQPVectorf  *rho_vec,
                               OSQPSettings       *settings,
                               c_int               polish);

/**
 * Solve linear system and store result in b
 * @param  s        Linear system solver structure
 * @param  b        Right-hand side
 * @return          Exitflag
 */
c_int solve_linsys_qdldl(qdldl_solver *s,
                         OSQPVectorf  *b,
                         c_int         admm_iter);


void warm_start_linsys_solver_qdldl(qdldl_solver      *s,
                                    const OSQPVectorf *x);


#if EMBEDDED != 1
/**
 * Update linear system solver matrices
 * @param  s        Linear system solver structure
 * @param  P        Matrix P
 * @param  A        Matrix A
 * @return          Exitflag
 */
c_int update_linsys_solver_matrices_qdldl(
                  qdldl_solver * s,
                  const OSQPMatrix *P,
                  const OSQPMatrix *A);




/**
 * Update rho_vec parameter in linear system solver structure
 * @param  s        Linear system solver structure
 * @param  rho_vec  new rho_vec value
 * @return          exitflag
 */
c_int update_linsys_solver_rho_vec_qdldl(qdldl_solver      *s,
                                         const OSQPVectorf *rho_vec,
                                         c_float            rho_sc);

#endif

#ifndef EMBEDDED
/**
 * Free linear system solver
 * @param s linear system solver object
 */
void free_linsys_solver_qdldl(qdldl_solver * s);
#endif

#ifdef __cplusplus
}
#endif

#endif
