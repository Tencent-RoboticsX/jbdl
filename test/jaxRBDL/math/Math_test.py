from typing import Mapping
import unittest
from test.support import EnvironmentVarGuard
import numpy as np
import math
from jaxRBDL.math import cross_matrix, cross_motion_space, cross_force_space, inverse_motion_space, spatial_transform
from jaxRBDL.math import Xrotx, Xroty, Xrotz, Xtrans
from oct2py import octave
import os


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OCTAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(CURRENT_PATH)), "octave")
MATH_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Math")
octave.addpath(MATH_PATH)


class TestMath(unittest.TestCase):
    def setUp(self):
        self.env = EnvironmentVarGuard()
        self.env.set('JAX_ENABLE_X64', '1')
        self.env.set('JAX_PLATFORM_NAME', 'cpu')  
    def test_cross_matrix(self):
        input = np.random.rand(*(3, 1))
        py_output = cross_matrix(input)
        oct_output = octave.CrossMatrix(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.rand(*(1, 3))
        py_output = cross_matrix(input)
        oct_output = octave.CrossMatrix(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.rand(*(3,))
        py_output = cross_matrix(input)
        oct_output = octave.CrossMatrix(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

    def test_cross_motion_space(self):
        input = np.random.rand(*(6, 1))
        py_output = cross_motion_space(input)
        oct_output = octave.CrossMotionSpace(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.rand(*(1, 6))
        py_output = cross_motion_space(input)
        oct_output = octave.CrossMotionSpace(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.rand(*(6,))
        py_output = cross_motion_space(input)
        oct_output = octave.CrossMotionSpace(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

    def test_cross_force_space(self):
        input = np.random.rand(*(6, 1))
        py_output = cross_force_space(input)
        oct_output = octave.CrossForceSpace(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.rand(*(1, 6))
        py_output = cross_force_space(input)
        oct_output = octave.CrossForceSpace(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.rand(*(6,))
        py_output = cross_force_space(input)
        oct_output = octave.CrossForceSpace(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)


    def test_Xrot(self):
        input = np.random.randn() * math.pi
        py_output = Xrotx(input)
        oct_output = octave.Xrotx(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertAlmostEqual(residual, 0.0, 5)

        py_output = Xroty(input)
        oct_output = octave.Xroty(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertAlmostEqual(residual, 0.0, 5)

        py_output = Xrotz(input)
        oct_output = octave.Xrotz(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertAlmostEqual(residual, 0.0, 5)

    def test_Xtrans(self):
        input = np.random.randn(*(3, 1))
        py_output = Xtrans(input)
        oct_output = octave.Xtrans(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.randn(*(1, 3))
        py_output = Xtrans(input)
        oct_output = octave.Xtrans(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

        input = np.random.randn(*(3,))
        py_output = Xtrans(input)
        oct_output = octave.Xtrans(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)

    def test_spatial_transform(self):
        input = (np.random.randn(*(3,3)), np.random.randn(*(3, 1)))
        py_output = spatial_transform(*input)
        oct_output = octave.SpatialTransform(*input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertAlmostEqual(residual, 0.0, 5)

        input = (np.random.randn(*(3,3)), np.random.randn(*(1, 3)))
        py_output = spatial_transform(*input)
        oct_output = octave.SpatialTransform(*input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertAlmostEqual(residual, 0.0, 5)

        input = (np.random.randn(*(3,3)), np.random.randn(*(3,)))
        py_output = spatial_transform(*input)
        oct_output = octave.SpatialTransform(*input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertAlmostEqual(residual, 0.0, 5)

    def test_inverse_motion_space(self):
        input = np.random.randn(*(6, 6))
        py_output = inverse_motion_space(input)
        oct_output = octave.InverseMotionSpace(input)
        residual = np.sum(np.abs(py_output - oct_output))
        self.assertEqual(residual, 0.0)
        

if __name__ == "__main__":
    unittest.main()
