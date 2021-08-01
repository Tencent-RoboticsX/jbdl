# Test cuosqp python module
import jbdl.experimental.cuosqp as osqp
from jbdl.experimental.cuosqp._osqp import constant
import numpy as np
from scipy import sparse

# Unit Test
import unittest
import numpy.testing as nptest


class non_convex_tests(unittest.TestCase):

    def setUp(self):

        # Simple QP problem
        self.P = sparse.triu([[2., 5.], [5., 1.]], format='csc')
        self.q = np.array([3, 4])
        self.A = sparse.csc_matrix([[-1.0, 0.], [0., -1.],
                                    [-1., 3.], [2., 5.], [3., 4]])
        self.u = np.array([0., 0., -15, 100, 80])
        self.l = -np.inf * np.ones(len(self.u))
        self.model = osqp.OSQP()

    def test_non_convex(self):
        # Setup workspace with new sigma
        opts = {'verbose': False, 'sigma': 5}
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **opts)

        # Solve problem
        res = self.model.solve()

        # Assert close
        self.assertEqual(res.info.status_val, constant('OSQP_NON_CVX'))
        nptest.assert_approx_equal(res.info.obj_val, np.nan)

