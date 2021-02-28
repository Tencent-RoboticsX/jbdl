import os
import jax
from jax.api import jit
from jax import grad
from oct2py import octave
import numpy as np
import math
import unittest
from test.support import EnvironmentVarGuard
from jaxRBDL.Dynamics.CompositeRigidBodyAlgorithm import CompositeRigidBodyAlgorithm, CompositeRigidBodyAlgorithmCore
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
from jaxRBDL.Dynamics.InverseDynamics import InverseDynamics, InverseDynamicsCore
from jaxRBDL.Utils.ModelWrapper import ModelWrapper
import time
import timeit


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
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_CompositeRigidBodyAlgorithm(self):
        input = (self.model, self.q * np.random.randn(*(7, )))
        py_output = CompositeRigidBodyAlgorithm(*input)
        oct_ouput = octave.CompositeRigidBodyAlgorithm(*input)
        self.assertAlmostEqual(np.sum(np.abs(py_output-oct_ouput)), 0.0, 5)

    def test_CompositeRigidBodyAlgorithmGradients(self):
        # print("Testing CompositeRigidBodyAlgorithmGradients!!!")
        def CompositeRigidBodyAlgorithmCoreWithJit():
            q = self.q * np.random.randn(*(7, ))
            input = (self.model["Xtree"], self.model["I"], tuple(self.model["parent"]), tuple(self.model["jtype"]), self.model["jaxis"],
                    self.model["NB"], q)
            CompositeRigidBodyAlgorithmCore(*input).block_until_ready()

        print(timeit.Timer(CompositeRigidBodyAlgorithmCoreWithJit).repeat(repeat=3, number=1000))
        

        q = self.q * np.random.randn(*(7, ))
        input = (self.model["Xtree"], self.model["I"], tuple(self.model["parent"]), tuple(self.model["jtype"]), self.model["jaxis"],
                self.model["NB"], q)
        fun1 = jit(jax.jacfwd(CompositeRigidBodyAlgorithmCore, argnums=(6,)), static_argnums=(2, 3, 4, 5))
        H2q, = fun1(*input)
        H2q.block_until_ready()



        def CompositeRigidBodyAlgorithmGradWithJit():
            q = self.q * np.random.randn(*(7, ))
            input = (self.model["Xtree"], self.model["I"], tuple(self.model["parent"]), tuple(self.model["jtype"]), self.model["jaxis"],
                    self.model["NB"], q)
            H2q, = fun1(*input)
            H2q.block_until_ready()

        print(timeit.Timer(CompositeRigidBodyAlgorithmGradWithJit).repeat(repeat=3, number=1000))

        q = self.q * np.random.randn(*(7, ))
        input = (self.model["Xtree"], self.model["I"], tuple(self.model["parent"]), tuple(self.model["jtype"]), self.model["jaxis"],
                self.model["NB"], q)
        fun2 = jax.jacfwd(CompositeRigidBodyAlgorithmCore, argnums=(6,))
        H2q, = fun2(*input)
        H2q.block_until_ready()

        def CompositeRigidBodyAlgorithmGradWithoutJit():
            q = self.q * np.random.randn(*(7, ))
            input = (self.model["Xtree"], self.model["I"], tuple(self.model["parent"]), tuple(self.model["jtype"]), self.model["jaxis"],
                    self.model["NB"], q)
            H2q, = fun2(*input)
            H2q.block_until_ready()
        
        print(timeit.Timer(CompositeRigidBodyAlgorithmGradWithoutJit).repeat(repeat=3, number=1))



        

        # H2Xtree,  H2I= jit(jax.jacfwd(CompositeRigidBodyAlgorithmCore, argnums=(0, 1)), static_argnums=(2, 3, 4, 5))(*input)
        # print("====================")
        # for item in H2Xtree:
        #     print(item.shape)

        # print("====================")
        # for item in H2I:
        #     print(item.shape)
            
        # print("====================")
        # H2q,  = jit(jax.jacfwd(CompositeRigidBodyAlgorithmCore, argnums=(6,)), static_argnums=(2, 3, 4, 5))(*input)
        # print(H2q.shape)



    def test_ForwardDynamics(self):
        q =  self.q * np.random.randn(*(7, ))
        qdot =  self.qdot * np.random.randn(*(7, ))
        tau = self.tau * np.random.randn(*(7, ))
        input = (self.model, q, qdot, tau)
        py_output = ForwardDynamics(*input)
        oct_output = octave.ForwardDynamics(*input)
        self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 3)

    def test_ForwardDynamicsGradients(self):
        # print("Testing ForwardDynamicsGradients!!!")
        q =  self.q * np.random.randn(*(7, ))
        qdot =  self.qdot * np.random.randn(*(7, ))
        tau = self.tau * np.random.randn(*(7, ))
        input = (self.model["Xtree"], self.model["I"], tuple(self.model["parent"]), tuple(self.model["jtype"]), self.model["jaxis"],
                 self.model["NB"], q, qdot, tau, self.model["a_grav"])
        ForwardDynamicsCore(*input)
        # qddot2Xtree,  qddot2I= jit(jax.jacfwd(ForwardDynamicsCore, argnums=(0, 1)), static_argnums=(2, 3, 4, 5))(*input)
        # print("====================")
        # for item in qddot2Xtree:
        #     print(item.shape)

        # print("====================")
        # for item in qddot2I:
        #     print(item.shape)
            
        # print("====================")
        # qddot2q, qddot2qdot, qddot2tau, qddot2a_grav = jit(jax.jacfwd(ForwardDynamicsCore, argnums=(6,7, 8, 9)), static_argnums=(2, 3, 4, 5))(*input)
        # print(qddot2q.shape)
        # print(qddot2qdot.shape)
        # print(qddot2tau.shape)
        # print(qddot2a_grav.shape)

    def test_InverseDynamics(self):
        q = self.q * np.random.randn(*(7, ))
        qdot = self.qdot * np.random.randn(*(7, ))
        qddot = self.qddot * np.random.randn(*(7, ))
        input = (self.model, q, qdot, qddot)
        oct_output = octave.InverseDynamics(*input)
        py_output = InverseDynamics(*input)
        self.assertAlmostEqual(np.sum(np.abs(py_output-oct_output)), 0.0, 4)

    def test_InverseDynamicsGradients(self):
        # print("Testing InverseDynamicsGradients!!!")
        q = self.q * np.random.randn(*(7, ))
        qdot = self.qdot * np.random.randn(*(7, ))
        qddot = self.qddot * np.random.randn(*(7, ))
        input = (self.model["Xtree"], self.model["I"], tuple(self.model["parent"]), tuple(self.model["jtype"]), self.model["jaxis"],
                 self.model["NB"], q, qdot, qddot, self.model["a_grav"])

        InverseDynamicsCore(*input)
        # tau2Xtree,  tau2I= jit(jax.jacfwd(InverseDynamicsCore, argnums=(0, 1)), static_argnums=(2, 3, 4, 5))(*input)
        # print("====================")
        # for item in tau2Xtree:
        #     print(item.shape)

        # print("====================")
        # for item in tau2I:
        #     print(item.shape)
            
        # print("====================")
        # tau2q, tau2qdot, tau2qddot, tau2a_grav = jit(jax.jacfwd(InverseDynamicsCore, argnums=(6,7, 8, 9)), static_argnums=(2, 3, 4, 5))(*input)
        # print(tau2q.shape)
        # print(tau2qdot.shape)
        # print(tau2qddot.shape)
        # print(tau2a_grav.shape)






if __name__ == "__main__":
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDynamics)
    unittest.TextTestRunner(verbosity=0).run(suite)
