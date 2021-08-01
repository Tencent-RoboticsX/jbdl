# Test cuosqp python module
import jbdl.experimental.cuosqp as osqp
from jbdl.experimental.cuosqp._osqp import constant
import numpy as np
from scipy import sparse

# Unit Test
import unittest


class dual_infeasibility_tests(unittest.TestCase):

    def setUp(self):
        """
        Setup default options
        """
        self.opts = {'verbose': False,
                     'eps_abs': 1e-05,
                     'eps_rel': 1e-05,
                     'eps_prim_inf': 1e-15,  # Focus only on dual infeasibility
                     'eps_dual_inf': 1e-6,
                     'scaling': 3,
                     'max_iter': 2500,
                     'polish': False,
                     'check_termination': 1,
                     'polish_refine_iter': 4}

    def test_dual_infeasible_lp(self):

        # Dual infeasible example
        self.P = sparse.csc_matrix((2, 2))
        self.q = np.array([2, -1])
        self.A = sparse.eye(2, format='csc')
        self.l = np.array([0., 0.])
        self.u = np.array([np.inf, np.inf])

        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

        # Solve problem with OSQP
        res = self.model.solve()

        # Assert close
        self.assertEqual(res.info.status_val, constant('OSQP_DUAL_INFEASIBLE'))

    def test_dual_infeasible_qp(self):

        # Dual infeasible example
        self.P = sparse.diags([4., 0.], format='csc')
        self.q = np.array([0, 2])
        self.A = sparse.csc_matrix([[1., 1.], [-1., 1.]])
        self.l = np.array([-np.inf, -np.inf])
        self.u = np.array([2., 3.])

        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

        # Solve problem with OSQP
        res = self.model.solve()

        # Assert close
        self.assertEqual(res.info.status_val, constant('OSQP_DUAL_INFEASIBLE'))

    def test_primal_and_dual_infeasible_problem(self):

        self.n = 2
        self.m = 4
        self.P = sparse.csc_matrix((2, 2))
        self.q = np.array([-1., -1.])
        self.A = sparse.csc_matrix([[1., -1.], [-1., 1.], [1., 0.], [0., 1.]])
        self.l = np.array([1., 1., 0., 0.])
        self.u = np.inf * np.ones(self.m)

        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

        # Warm start to avoid infeasibility detection at first step
        x0 = 25.*np.ones(self.n)
        y0 = -2.*np.ones(self.m)
        self.model.warm_start(x=x0, y=y0)

        # Solve
        res = self.model.solve()

        # Assert close
        self.assertIn(res.info.status_val, [constant('OSQP_PRIMAL_INFEASIBLE'),
                                            constant('OSQP_DUAL_INFEASIBLE')])
