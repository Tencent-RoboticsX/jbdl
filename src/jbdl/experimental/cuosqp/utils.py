"""Common utility functions"""
from warnings import warn
import numpy as np
import scipy.sparse as sparse
from jbdl.experimental.cuosqp import _osqp


def linsys_solver_str_to_int(settings):
    linsys_solver_str = settings.pop('linsys_solver', '')
    if not isinstance(linsys_solver_str, str):
        raise TypeError("Setting linsys_solver " +
                        "is required to be a string.")
    linsys_solver_str = linsys_solver_str.lower()
    if linsys_solver_str == 'cuda pcg':
        settings['linsys_solver'] = _osqp.constant('CUDA_PCG_SOLVER')
    # Default solver: CUDA PCG
    elif linsys_solver_str == '':
        settings['linsys_solver'] = _osqp.constant('CUDA_PCG_SOLVER')
    else:   # default solver: CUDA PCG
        warn("Linear system solver not recognized. " +
             "Using default solver CUDA PCG.")
        settings['linsys_solver'] = _osqp.constant('CUDA_PCG_SOLVER')
    return settings


def prepare_data(p_matrix=None, q=None, a_matrix=None, l=None, u=None, **settings):
    """
    Prepare problem data of the form

    minimize     1/2 x' * p_matrix * x + q' * x
    subject to   l <= a_matrix * x <= u

    solver settings can be specified as additional keyword arguments
    """

    #
    # Get problem dimensions
    #

    if p_matrix is None:
        if q is not None:
            n = len(q)
        elif a_matrix is not None:
            n = a_matrix.shape[1]
        else:
            raise ValueError("The problem does not have any variables")
    else:
        n = p_matrix.shape[0]
    if a_matrix is None:
        m = 0
    else:
        m = a_matrix.shape[0]

    #
    # Create parameters if they are None
    #

    if (a_matrix is None and (l is not None or u is not None)) or \
            (a_matrix is not None and (l is None and u is None)):
        raise ValueError("a_matrix must be supplied together " +
                         "with at least one bound l or u")

    # Add infinity bounds in case they are not specified
    if a_matrix is not None and l is None:
        l = -np.inf * np.ones(a_matrix.shape[0])
    if a_matrix is not None and u is None:
        u = np.inf * np.ones(a_matrix.shape[0])

    # Create elements if they are not specified
    if p_matrix is None:
        p_matrix = sparse.csc_matrix((np.zeros((0,), dtype=np.double),
                               np.zeros((0,), dtype=np.int),
                               np.zeros((n+1,), dtype=np.int)),
                              shape=(n, n))
    if q is None:
        q = np.zeros(n)

    if a_matrix is None:
        a_matrix = sparse.csc_matrix((np.zeros((0,), dtype=np.double),
                               np.zeros((0,), dtype=np.int),
                               np.zeros((n+1,), dtype=np.int)),
                              shape=(m, n))
        l = np.zeros(a_matrix.shape[0])
        u = np.zeros(a_matrix.shape[0])

    #
    # Check vector dimensions (not checked from C solver)
    #

    # Check if second dimension of a_matrix is correct
    # if a_matrix.shape[1] != n:
    #     raise ValueError("Dimension n in a_matrix and p_matrix does not match")
    if len(q) != n:
        raise ValueError("Incorrect dimension of q")
    if len(l) != m:
        raise ValueError("Incorrect dimension of l")
    if len(u) != m:
        raise ValueError("Incorrect dimension of u")

    #
    # Check or Sparsify Matrices
    #
    if not sparse.issparse(p_matrix) and isinstance(p_matrix, np.ndarray) and \
            len(p_matrix.shape) == 2:
        raise TypeError("p_matrix is required to be a sparse matrix")
    if not sparse.issparse(a_matrix) and isinstance(a_matrix, np.ndarray) and \
            len(a_matrix.shape) == 2:
        raise TypeError("a_matrix is required to be a sparse matrix")

    # If p_matrix is not triu, then convert it to triu
    if sparse.tril(p_matrix, -1).data.size > 0:
        p_matrix = sparse.triu(p_matrix, format='csc')

    # Convert matrices in CSC form and to individual pointers
    if not sparse.isspmatrix_csc(p_matrix):
        warn("Converting sparse p_matrix to a CSC " +
             "(compressed sparse column) matrix. (It may take a while...)")
        p_matrix = p_matrix.tocsc()
    if not sparse.isspmatrix_csc(a_matrix):
        warn("Converting sparse a_matrix to a CSC " +
             "(compressed sparse column) matrix. (It may take a while...)")
        a_matrix = a_matrix.tocsc()

    # Check if p_matrix an a_matrix have sorted indices
    if not p_matrix.has_sorted_indices:
        p_matrix.sort_indices()
    if not a_matrix.has_sorted_indices:
        a_matrix.sort_indices()

    # Convert infinity values to OSQP Infinity
    u = np.minimum(u, _osqp.constant('OSQP_INFTY'))
    l = np.maximum(l, -_osqp.constant('OSQP_INFTY'))

    # Convert linsys_solver string to integer
    settings = linsys_solver_str_to_int(settings)

    return ((n, m), p_matrix.data, p_matrix.indices, p_matrix.indptr, q,
            a_matrix.data, a_matrix.indices, a_matrix.indptr,
            l, u), settings
