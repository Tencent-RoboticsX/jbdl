"""
Python interface module for OSQP solver
"""
from builtins import object
from jbdl.experimental.cuosqp import _osqp
import numpy as np
from platform import system
from jbdl.experimental.cuosqp import utils
import sys


class OSQP(object):
    def __init__(self):
        self._model = _osqp.OSQP()

    def version(self):
        return self._model.version()

    def setup(self, P=None, q=None, A=None, l=None, u=None, **settings):
        """
        Setup OSQP solver problem of the form

        minimize     1/2 x' * P * x + q' * x
        subject to   l <= A * x <= u

        solver settings can be specified as additional keyword arguments
        """

        unpacked_data, settings = utils.prepare_data(P, q, A, l, u, **settings)
        self._model.setup(*unpacked_data, **settings)

    def update(self, q=None, l=None, u=None,
               Px=None, Px_idx=np.array([]), Ax=None, Ax_idx=np.array([])):
        """
        Update OSQP problem arguments
        """

        # get problem dimensions
        (n, m) = self._model.dimensions()

        # check consistency of the input arguments
        if q is not None and len(q) != n:
            raise ValueError("q must have length n")
        if l is not None:
            if not isinstance(l, np.ndarray):
                raise TypeError("l must be numpy.ndarray, not %s" %
                                type(l).__name__)
            elif len(l) != m:
                raise ValueError("l must have length m")
            # Convert values to -OSQP_INFTY
            l = np.maximum(l, -_osqp.constant('OSQP_INFTY'))
        if u is not None:
            if not isinstance(u, np.ndarray):
                raise TypeError("u must be numpy.ndarray, not %s" %
                                type(u).__name__)
            elif len(u) != m:
                raise ValueError("u must have length m")
            # Convert values to OSQP_INFTY
            u = np.minimum(u, _osqp.constant('OSQP_INFTY'))
        if Ax is None:
            if len(Ax_idx) > 0:
                raise ValueError("Vector Ax has not been specified")
        else:
            if len(Ax_idx) > 0 and len(Ax) != len(Ax_idx):
                raise ValueError("Ax and Ax_idx must have the same lengths")
        if Px is None:
            if len(Px_idx) > 0:
                raise ValueError("Vector Px has not been specified")
        else:
            if len(Px_idx) > 0 and len(Px) != len(Px_idx):
                raise ValueError("Px and Px_idx must have the same lengths")
        if q is None and l is None and u is None and Px is None and Ax is None:
            raise ValueError("No updatable data has been specified")

        # update linear cost
        if q is not None:
            self._model.update_lin_cost(q)

        # update lower bound
        if l is not None and u is None:
            self._model.update_bounds(l, np.array([]))

        # update upper bound
        if u is not None and l is None:
            self._model.update_bounds(np.array([]), u)

        # update bounds
        if l is not None and u is not None:
            self._model.update_bounds(l, u)

        # update matrix P
        if Px is not None and Ax is None:
            self._model.update_P(Px, Px_idx, len(Px))

        # update matrix A
        if Ax is not None and Px is None:
            self._model.update_A(Ax, Ax_idx, len(Ax))

        # update matrices P and A
        if Px is not None and Ax is not None:
            self._model.update_P_A(Px, Px_idx, len(Px), Ax, Ax_idx, len(Ax))

    def update_settings(self, **kwargs):
        """
        Update OSQP solver settings

        It is possible to change: 'max_iter', 'eps_abs', 'eps_rel',
                                  'eps_prim_inf', 'eps_dual_inf', 'rho'
                                  'alpha', 'delta', 'polish',
                                  'polish_refine_iter',
                                  'verbose', 'scaled_termination',
                                  'check_termination', 'time_limit',
        """

        # get arguments
        max_iter = kwargs.pop('max_iter', None)
        eps_abs = kwargs.pop('eps_abs', None)
        eps_rel = kwargs.pop('eps_rel', None)
        eps_prim_inf = kwargs.pop('eps_prim_inf', None)
        eps_dual_inf = kwargs.pop('eps_dual_inf', None)
        rho = kwargs.pop('rho', None)
        alpha = kwargs.pop('alpha', None)
        delta = kwargs.pop('delta', None)
        polish = kwargs.pop('polish', None)
        polish_refine_iter = kwargs.pop('polish_refine_iter', None)
        verbose = kwargs.pop('verbose', None)
        scaled_termination = kwargs.pop('scaled_termination', None)
        check_termination = kwargs.pop('check_termination', None)
        warm_start = kwargs.pop('warm_start', None)
        time_limit = kwargs.pop('time_limit', None)

        # update them
        if max_iter is not None:
            self._model.update_max_iter(max_iter)

        if eps_abs is not None:
            self._model.update_eps_abs(eps_abs)

        if eps_rel is not None:
            self._model.update_eps_rel(eps_rel)

        if eps_prim_inf is not None:
            self._model.update_eps_prim_inf(eps_prim_inf)

        if eps_dual_inf is not None:
            self._model.update_eps_dual_inf(eps_dual_inf)

        if rho is not None:
            self._model.update_rho(rho)

        if alpha is not None:
            self._model.update_alpha(alpha)

        if delta is not None:
            self._model.update_delta(delta)

        if polish is not None:
            self._model.update_polish(polish)

        if polish_refine_iter is not None:
            self._model.update_polish_refine_iter(polish_refine_iter)

        if verbose is not None:
            self._model.update_verbose(verbose)

        if scaled_termination is not None:
            self._model.update_scaled_termination(scaled_termination)

        if check_termination is not None:
            self._model.update_check_termination(check_termination)

        if warm_start is not None:
            self._model.update_warm_start(warm_start)

        if time_limit is not None:
            self._model.update_time_limit(time_limit)

        if max_iter is None and \
           eps_abs is None and \
           eps_rel is None and \
           eps_prim_inf is None and \
           eps_dual_inf is None and \
           rho is None and \
           alpha is None and \
           delta is None and \
           polish is None and \
           polish_refine_iter is None and \
           verbose is None and \
           scaled_termination is None and \
           check_termination is None and \
           warm_start is None:
            raise ValueError("No updatable settings has been specified!")

    def solve(self):
        """
        Solve QP Problem
        """
        # Solve QP
        return self._model.solve()

    def warm_start(self, x=None, y=None):
        """
        Warm start primal or dual variables
        """
        # get problem dimensions
        (n, m) = self._model.dimensions()

        if x is not None:
            if len(x) != n:
                raise ValueError("Wrong dimension for variable x")

            if y is None:
                self._model.warm_start(x, np.array([]))

        if y is not None:
            if len(y) != m:
                raise ValueError("Wrong dimension for variable y")

            if x is None:
                self._model.warm_start(np.array([]), y)

        if x is not None and y is not None:
            self._model.warm_start(x, y)

        if x is None and y is None:
            raise ValueError("Unrecognized fields")
