import unittest
from test.support import EnvironmentVarGuard
import os
from oct2py import octave
import math
import numpy as np
from jaxBDL.rbdl.model import rigid_body_inertia


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OCTAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(CURRENT_PATH)), "octave")
MATH_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Math")
MODEL_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Model")
octave.addpath(MATH_PATH)
octave.addpath(MODEL_PATH)

class TestRigidBodyInertia(unittest.TestCase):
    def setUp(self):
        self.env = EnvironmentVarGuard()
        self.env.set('JAX_ENABLE_X64', '1')
        self.env.set('JAX_PLATFORM_NAME', 'cpu')  
    def test_rigid_body_inertia(self):
        input = (np.random.randn(), np.random.randn(*(3, 1)), np.random.randn(*(3, 3)))
        py_output = rigid_body_inertia(*input)
        oct_output = octave.RigidBodyInertia(*input)
        self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 5)

        input = (np.random.randn(), np.random.randn(*(1, 3)), np.random.randn(*(3, 3)))
        py_output = rigid_body_inertia(*input)
        oct_output = octave.RigidBodyInertia(*input)
        self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 5)

        input = (np.random.randn(), np.random.randn(*(3,)), np.random.randn(*(3, 3)))
        py_output = rigid_body_inertia(*input)
        oct_output = octave.RigidBodyInertia(*input)
        self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 5)

if __name__ == "__main__":
    unittest.main()