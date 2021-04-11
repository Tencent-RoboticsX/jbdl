import unittest
import os
from oct2py import octave
import math
import numpy as np
from jaxRBDL.Model import joint_model
from test.support import EnvironmentVarGuard


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OCTAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(CURRENT_PATH)), "octave")
MATH_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Math")
MODEL_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Model")
octave.addpath(MATH_PATH)
octave.addpath(MODEL_PATH)

class TestJointModel(unittest.TestCase):
    def setUp(self):
        self.env = EnvironmentVarGuard()
        self.env.set('JAX_ENABLE_X64', '1')
        self.env.set('JAX_PLATFORM_NAME', 'cpu')  
    def test_joint_model(self):
        jtype_list = [0, 1]
        jaxis_list = ['x', 'y', 'z'] 

        for jtype in jtype_list:
            for jaxis in jaxis_list:
                input = (jtype, jaxis, math.pi * np.random.rand())
                py_output = joint_model(*input)
                oct_output = joint_model(*input)
                for py_elem, oct_elem in zip(py_output, oct_output):
                    self.assertEqual(np.sum(np.abs(py_elem-oct_elem)), 0.0)

if __name__ == "__main__":
    unittest.main()