import unittest
import os
from oct2py import octave
import numpy as np
from jaxRBDL.Kinematics.TransformToPosition import TransformToPosition

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OCTAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(CURRENT_PATH)), "octave")
MATH_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Math")
KINEMATICS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Kinematics")
octave.addpath(MATH_PATH)
octave.addpath(KINEMATICS_PATH)

class TestTransformToPosition(unittest.TestCase):
    def test_TransformToPosition(self):
        input = np.random.randn(*(6, 6))
        py_output = TransformToPosition(input)
        oct_output = octave.TransformToPosition(input)
        self.assertAlmostEqual(np.sum(np.abs(py_output - oct_output)), 0.0, 5)

if __name__ == "__main__":
    unittest.main()