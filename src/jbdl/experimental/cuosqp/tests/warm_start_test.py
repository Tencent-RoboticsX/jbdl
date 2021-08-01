# Test cuosqp python module
import jbdl.experimental.cuosqp as osqp
import numpy as np
from scipy import sparse
from numpy.random import Generator, PCG64

# Unit Test
import unittest


class warm_start_tests(unittest.TestCase):

  def setUp(self):
    """
    Setup default options
    """
    self.opts = {'verbose': False,
                 'adaptive_rho': False,
                 'eps_abs': 1e-05,
                 'eps_rel': 1e-05,
                 'polish': False,
                 'check_termination': 1}

  def test_warm_start(self):
    # Set random seed for reproducibility
    rg = Generator(PCG64(1))

    self.n = 100
    self.m = 200
    self.A = sparse.random(self.m, self.n, density=0.9, format='csc', random_state=rg)
    self.l = -rg.random(self.m) * 2.
    self.u = rg.random(self.m) * 2.

    P = sparse.random(self.n, self.n, density=0.9, random_state=rg)
    self.P = sparse.triu(P.dot(P.T), format='csc')
    self.q = rg.standard_normal(self.n)

    # Setup solver
    self.model = osqp.OSQP()
    self.model.setup(self.P, self.q, self.A, self.l, self.u, **self.opts)

    # Solve problem with OSQP
    res = self.model.solve()

    # Store optimal values
    x_opt = res.x
    y_opt = res.y
    tot_iter = res.info.iter

    # Warm start with zeros and check if number of iterations is the same
    self.model.warm_start(x=np.zeros(self.n), y=np.zeros(self.m))
    res = self.model.solve()
    self.assertEqual(res.info.iter, tot_iter)

    # Warm start with optimal values and check that number of iter < 10
    self.model.warm_start(x=x_opt, y=y_opt)
    res = self.model.solve()
    self.assertLess(res.info.iter, 10)
