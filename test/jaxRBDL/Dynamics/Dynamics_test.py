import os
import jax
from oct2py import octave
import numpy as np
import math
import unittest
from test.support import EnvironmentVarGuard
from jaxRBDL.Dynamics.CompositeRigidBodyAlgorithm import CompositeRigidBodyAlgorithm
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics
from jaxRBDL.Dynamics.InverseDynamics import InverseDynamics
from jaxRBDL.Utils.ModelWrapper import ModelWrapper


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OCTAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(CURRENT_PATH)), "octave")
MRBDL_PATH = os.path.join(OCTAVE_PATH, "mRBDL")
MATH_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Math")
MODEL_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Model")
TOOLS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Tools")
KINEMATICS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Kinematics")
DYNAMICS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Dynamics")
IPPHYSICIALPARAMS_PATH = os.path.join(OCTAVE_PATH, "ipPhysicalParams") 
IPTOOLS_PATH = os.path.join(OCTAVE_PATH, "ipTools")


octave.addpath(MRBDL_PATH)
octave.addpath(MATH_PATH)
octave.addpath(MODEL_PATH)
octave.addpath(TOOLS_PATH)
octave.addpath(KINEMATICS_PATH)
octave.addpath(DYNAMICS_PATH)
octave.addpath(IPPHYSICIALPARAMS_PATH)
octave.addpath(IPTOOLS_PATH)
octave.addpath(OCTAVE_PATH)


class TestDynamics(unittest.TestCase):
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
        self.tau = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.env = EnvironmentVarGuard()
        self.env.set('JAX_ENABLE_X64', '1')
        self.env.set('JAX_PLATFORM_NAME', 'cpu')  

    def test_CompositeRigidBodyAlgorithm(self):
        input = (self.model, self.q * np.random.randn(*(7, )))
        py_output = CompositeRigidBodyAlgorithm(*input)
        oct_ouput = octave.CompositeRigidBodyAlgorithm(*input)
        self.assertAlmostEqual(np.sum(np.abs(py_output-oct_ouput)), 0.0, 5)

    def test_ForwardDynamics(self):
        q =  self.q * np.random.randn(*(7, ))
        qdot =  self.qdot * np.random.randn(*(7, ))
        tau = self.tau * np.random.randn(*(7, ))
        input = (self.model, q, qdot, tau)
        py_output = ForwardDynamics(*input)
        oct_output = octave.ForwardDynamics(*input)
        self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 4)

    def test_InverseDynamics(self):
        q = self.q * np.random.randn(*(7, ))
        qdot = self.qdot * np.random.randn(*(7, ))
        qddot = self.qddot * np.random.randn(*(7, ))
        input = (self.model, q, qdot, qddot)
        oct_output = octave.InverseDynamics(*input)
        py_output = InverseDynamics(*input)
        self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 4)




if __name__ == "__main__":
    unittest.main()
