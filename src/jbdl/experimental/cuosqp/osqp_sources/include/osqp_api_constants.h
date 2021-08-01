#ifndef OSQP_API_CONSTANTS_H
#define OSQP_API_CONSTANTS_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

#include "osqp_configure.h"

/*******************
* OSQP Versioning *
*******************/
# define OSQP_VERSION ("0.6.0") /* string literals automatically null-terminated
                                   */

/******************
* Solver Status  *
******************/
# define OSQP_DUAL_INFEASIBLE_INACCURATE (4)
# define OSQP_PRIMAL_INFEASIBLE_INACCURATE (3)
# define OSQP_SOLVED_INACCURATE (2)
# define OSQP_SOLVED (1)
# define OSQP_MAX_ITER_REACHED (-2)
# define OSQP_PRIMAL_INFEASIBLE (-3)    /* primal infeasible  */
# define OSQP_DUAL_INFEASIBLE (-4)      /* dual infeasible */
# define OSQP_SIGINT (-5)               /* interrupted by user */
# ifdef PROFILING
#  define OSQP_TIME_LIMIT_REACHED (-6)
# endif // ifdef PROFILING
# define OSQP_NON_CVX (-7)              /* problem non convex */
# define OSQP_UNSOLVED (-10)            /* Unsolved. Only setup function has been called */


/*************************
* Linear System Solvers *
*************************/
enum linsys_solver_type { QDLDL_SOLVER, MKL_PARDISO_SOLVER, CUDA_PCG_SOLVER };
extern const char * LINSYS_SOLVER_NAME[];


/******************
* Solver Errors  *
******************/
enum osqp_error_type {
    OSQP_DATA_VALIDATION_ERROR = 1,  /* Start errors from 1 */
    OSQP_SETTINGS_VALIDATION_ERROR,
    OSQP_LINSYS_SOLVER_LOAD_ERROR,
    OSQP_LINSYS_SOLVER_INIT_ERROR,
    OSQP_NONCVX_ERROR,
    OSQP_MEM_ALLOC_ERROR,
    OSQP_WORKSPACE_NOT_INIT_ERROR,
    OSQP_ALGEBRA_LOAD_ERROR
};
extern const char * OSQP_ERROR_MESSAGE[];


/**********************************
* Solver Parameters and Settings *
**********************************/

# define RHO (0.1)
# define SIGMA (1E-06)
# define MAX_ITER (4000)
# define EPS_ABS (1E-3)
# define EPS_REL (1E-3)
# define EPS_PRIM_INF (1E-4)
# define EPS_DUAL_INF (1E-4)
# define ALPHA (1.6)

#ifdef CUDA_SUPPORT
# define LINSYS_SOLVER (CUDA_PCG_SOLVER)
#else
# define LINSYS_SOLVER (QDLDL_SOLVER)
#endif

# define RHO_MIN (1e-06)
# define RHO_MAX (1e06)
# define RHO_TOL (1e-04) ///< tolerance for detecting if an inequality is set to equality
# define RHO_EQ_OVER_RHO_INEQ (1e03)

#ifdef CUDA_SUPPORT
#  define RHO_IS_VEC (0)
#else
#  define RHO_IS_VEC (1)  ///< boolean, defines if rho is scalar or vector
#endif

# ifndef EMBEDDED
#  define DELTA (1E-6)
#  define POLISH (0)
#  define POLISH_REFINE_ITER (3)
#  define VERBOSE (0)
# endif // ifndef EMBEDDED

#ifdef CUDA_SUPPORT
#  define CHECK_TERMINATION (5)
#else
#  define CHECK_TERMINATION (25)
#endif

# define SCALED_TERMINATION (0)
# define WARM_START (1)
# define SCALING (10)

# define MIN_SCALING (1e-04) ///< minimum scaling value
# define MAX_SCALING (1e+04) ///< maximum scaling value


# ifndef OSQP_NULL
#  define OSQP_NULL 0
# endif /* ifndef OSQP_NULL */

# ifndef OSQP_NAN
#  define OSQP_NAN ((c_float)0x7fc00000UL)  // not a number
# endif /* ifndef OSQP_NAN */

# ifndef OSQP_INFTY
#if defined(CUDA_SUPPORT) && defined(DFLOAT)
// Multiplying two floats that are in the order of 1e20 results in an overflow
#  define OSQP_INFTY ((c_float)1e17)
#else
#  define OSQP_INFTY ((c_float)1e20)        // infinity
#endif
# endif /* ifndef OSQP_INFTY */


# if EMBEDDED != 1
#  define ADAPTIVE_RHO (1)

#ifdef CUDA_SUPPORT
#  define ADAPTIVE_RHO_INTERVAL (10)
#  define ADAPTIVE_RHO_TOLERANCE (2.0)
#else
#  define ADAPTIVE_RHO_INTERVAL (0)
#  define ADAPTIVE_RHO_TOLERANCE (5.0)          ///< tolerance for adopting new rho; minimum ratio between new rho and the current one
#endif

#  define ADAPTIVE_RHO_FRACTION (0.4)           ///< fraction of setup time after which we update rho
#  define ADAPTIVE_RHO_MULTIPLE_TERMINATION (4) ///< multiple of check_termination after which we update rho (if PROFILING disabled)
#  define ADAPTIVE_RHO_FIXED (100)              ///< number of iterations after which we update rho if termination_check  and PROFILING are disabled
# endif // if EMBEDDED != 1

# ifdef PROFILING
#  define TIME_LIMIT (0)                        ///< Disable time limit as default
# endif // ifdef PROFILING

/* Printing */
# define PRINT_INTERVAL 200


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef OSQP_API_CONSTANTS_H
