import os
from oct2py import octave
import numpy as np
import math
import unittest
from test.support import EnvironmentVarGuard
from jaxRBDL.Kinematics import calc_body_to_base_coordinates
from jaxRBDL.Kinematics.CalcPointVelocity import CalcPointVelocity
from jaxRBDL.Kinematics.CalcPointVelocity import CalcPointVelocityCore
from jaxRBDL.Kinematics.CalcPointAcceleraion import CalcPointAcceleration
from jaxRBDL.Kinematics.CalcPointAcceleraion import CalcPointAccelerationCore
from jaxRBDL.Kinematics.calc_point_jacobian import calc_point_jacobian, calc_point_jacobian_core
from jaxRBDL.Kinematics.CalcPointJacobianDerivative import CalcPointJacobianDerivative
from jaxRBDL.Kinematics.CalcPointJacobianDerivative import CalcPointJacobianDerivativeCore
from jaxRBDL.Utils.ModelWrapper import ModelWrapper


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OCTAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(CURRENT_PATH)), "octave")
MRBDL_PATH = os.path.join(OCTAVE_PATH, "mRBDL")
MATH_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Math")
MODEL_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Model")
TOOLS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Tools")
KINEMATICS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Kinematics")
IPPHYSICIALPARAMS_PATH = os.path.join(OCTAVE_PATH, "ipPhysicalParams") 
IPTOOLS_PATH = os.path.join(OCTAVE_PATH, "ipTools")


octave.addpath(MRBDL_PATH)
octave.addpath(MATH_PATH)
octave.addpath(MODEL_PATH)
octave.addpath(TOOLS_PATH)
octave.addpath(KINEMATICS_PATH)
octave.addpath(IPPHYSICIALPARAMS_PATH)
octave.addpath(IPTOOLS_PATH)
octave.addpath(OCTAVE_PATH)


class TestKinematics(unittest.TestCase):
    def setUp(self):
        ip = dict()
        model = dict()
        octave.push("ip", ip)
        octave.push("model", model)
        self.ip = octave.ipParmsInit(0, 0, 0, 0)
   
        self.model = ModelWrapper(octave.model_create()).model

        self.q = np.array([0.0,  0.4765, 0, math.pi/6, math.pi/6, -math.pi/3, -math.pi/3])
        self.qdot = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.qddot = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.env = EnvironmentVarGuard()
        self.env.set('JAX_ENABLE_X64', '1')
        self.env.set('JAX_PLATFORM_NAME', 'cpu')  

    def test_CalcBodyToBaseCoordinates(self):
        for i in range(1, int(self.model['NB']) + 1):
            input = (self.model, self.q, i, np.random.rand(*(3,)))
            py_output = calc_body_to_base_coordinates(*input)
            oct_output = octave.CalcBodyToBaseCoordinates(*input)
            self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 5)
        
    
    def test_CalcPointVelocity(self):
        for i in range(1, int(self.model['NB']) + 1):
            input = (self.model, self.q, self.qdot, i, np.random.rand(*(3,)))
            py_output = CalcPointVelocity(*input)
            oct_output =  octave.CalcPointVelocity(*input)
            self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 5)
 

    def test_CalcPointAcceleration(self):
        for i in range(1, int(self.model['NB']) + 1):
            input = (self.model, self.q, self.qdot, self.qddot, i, np.random.rand(*(3,)))
            py_output = CalcPointAcceleration(*input)
            oct_output = octave.CalcPointAcceleration(*input)
            self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 5)


    def test_CalcPointJacobian(self):
        for i in range(1, int(self.model['NB']) + 1):
            input = (self.model, self.q, i, np.random.rand(*(3,)))
            oct_output = octave.CalcPointJacobian(*input)
            py_output = calc_point_jacobian(*input)
            self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 5)


        
    def test_CalcPointJacobianDerivative(self):
        for i in range(1, int(self.model['NB']) + 1):
            input = (self.model, self.q, self.qdot, i, np.random.rand(*(3,)))
            py_output = CalcPointJacobianDerivative(*input)
            oct_output = octave.CalcPointJacobianDerivative(*input)
            self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 5)

     




if __name__ == "__main__":
    unittest.main()
