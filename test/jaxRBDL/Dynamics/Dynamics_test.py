import os
import jax
from jax.api import jit
from jax import grad
# from oct2py import octave
import numpy as np
import math
import unittest
from numpy.core.fromnumeric import shape
from test.support import EnvironmentVarGuard
from jaxRBDL.Dynamics.CompositeRigidBodyAlgorithm import CompositeRigidBodyAlgorithm, CompositeRigidBodyAlgorithmCore
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
from jaxRBDL.Dynamics.InverseDynamics import InverseDynamics, InverseDynamicsCore
from jaxRBDL.Dynamics.StateFunODE import DynamicsFunCore
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


# octave.addpath(MRBDL_PATH)
# octave.addpath(MATH_PATH)
# octave.addpath(MODEL_PATH)
# octave.addpath(TOOLS_PATH)
# octave.addpath(KINEMATICS_PATH)
# octave.addpath(DYNAMICS_PATH)
# octave.addpath(IPPHYSICIALPARAMS_PATH)
# octave.addpath(IPTOOLS_PATH)
# octave.addpath(OCTAVE_PATH)

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(CURRENT_PATH), "Data")
print(DATA_PATH)



class TestDynamics(unittest.TestCase):
    def setUp(self):
        mdlw = ModelWrapper()
        mdlw.load(os.path.join(DATA_PATH, 'whole_max_v1.json'))
        self.model = mdlw.model
        self.q = np.array([
            0, 0, 0.27, 0, 0, 0, # base
            0, 0.5, -0.8,  # fr
            0, 0.5, -0.8,  # fl
            0, 0.5, -0.8,  # br
            0, 0.5, -0.8]) # bl
        self.NB = self.model["NB"]
        self.NC = self.model["NC"]
        self.qdot = np.ones(self.NB)
        self.qddot = np.ones(self.NB)
        self.tau = np.concatenate([np.zeros(6), np.ones(self.NB-6)])
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_DynamicsFunCore(self):
        model = self.model
        NC = int(model["NC"])
        NB = int(model["NB"])
        nf = int(model["nf"])
        Xtree = model["Xtree"]
        contactpoint = model["contactpoint"],
        idcontact = tuple(model["idcontact"])
        parent = tuple(model["parent"])
        jtype = tuple(model["jtype"])
        jaxis = model["jaxis"]
        contactpoint = model["contactpoint"]
        flag_contact = (1, 1, 1, 1)
        I = model["I"]
        q = self.q
        qdot = self.qdot
        tau = self.tau
        a_grav = model["a_grav"]
        rankJc = int(np.sum( [1 for item in flag_contact if item != 0]) * model["nf"])

        input = (Xtree, I, q, qdot, contactpoint, tau, a_grav, \
            idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, rankJc)

        print(DynamicsFunCore(*input))

        def DynamicsFunCoreWithJit():
            q = self.q * np.random.randn(*self.q.shape)
            qdot =  self.qdot * np.random.randn(*self.qdot.shape)
            input = (Xtree, I, q, qdot, contactpoint, tau, a_grav, \
                idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, rankJc)
            DynamicsFunCore(*input)

        print(timeit.Timer(DynamicsFunCoreWithJit).repeat(repeat=3, number=1000))

        
        # from jax import make_jaxpr
        # print(make_jaxpr(DynamicsFunCore, static_argnums=(7, 8, 9, 10, 11, 12, 13, 14, 15))(*input))
    # def test_CompositeRigidBodyAlgorithm(self):
    #     input = (self.model, self.q)
    #     CompositeRigidBodyAlgorithm(*input)

    #     def CompositeRigidBodyAlgorithmWithJit():
    #         input =  (self.model, self.q * np.random.randn(*self.q.shape))
    #         CompositeRigidBodyAlgorithm(*input)
    #     print("CompositeRigidBodyAlgorithm:")
    #     print(timeit.Timer(CompositeRigidBodyAlgorithmWithJit).repeat(repeat=3, number=1000))


    # def test_CompositeRigidBodyAlgorithmGradients(self):
    #     q = self.q * np.random.randn(*self.q.shape)
    #     input = (self.model["Xtree"], self.model["I"], tuple(self.model["parent"]), tuple(self.model["jtype"]), self.model["jaxis"],
    #             self.model["NB"], q)
    #     fun1 = jit(jax.jacfwd(CompositeRigidBodyAlgorithmCore, argnums=(6,)), static_argnums=(2, 3, 4, 5))
    #     H2q, = fun1(*input)
    #     H2q.block_until_ready()



    #     def CompositeRigidBodyAlgorithmGradWithJit():
    #         q = self.q * np.random.randn(*self.q.shape)
    #         input = (self.model["Xtree"], self.model["I"], tuple(self.model["parent"]), tuple(self.model["jtype"]), self.model["jaxis"],
    #                 self.model["NB"], q)
    #         H2q, = fun1(*input)
    #         H2q.block_until_ready()
    #     print("CompositeRigidBodyAlgorithmGrad:")
    #     print(timeit.Timer(CompositeRigidBodyAlgorithmGradWithJit).repeat(repeat=3, number=1000))

   


    # def test_ForwardDynamics(self):
    #     q =  self.q
    #     qdot =  self.qdot 
    #     tau = self.tau
    #     input = (self.model, q, qdot, tau)
    #     ForwardDynamics(*input)

    #     def ForwardDynamicsWithJit():
    #         input = (self.model, q * np.random.randn(*q.shape), qdot * np.random.randn(*qdot.shape), tau * np.random.randn(*tau.shape))
    #         ForwardDynamics(*input)

    #     print("ForwardDynamics:")
    #     print(timeit.Timer(ForwardDynamicsWithJit).repeat(repeat=3, number=1000))




    # def test_ForwardDynamicsGradients(self):
    #     q =  self.q * np.random.randn(*self.q.shape)
    #     qdot =  self.qdot * np.random.randn(*self.qddot.shape)
    #     tau = self.tau * np.random.randn(*self.tau.shape)
    #     input = (self.model["Xtree"], self.model["I"], tuple(self.model["parent"]), tuple(self.model["jtype"]), self.model["jaxis"],
    #              self.model["NB"], q, qdot, tau, self.model["a_grav"])
    #     ForwardDynamicsCore(*input)
        
    # def test_InverseDynamics(self):
    #     q = self.q * np.random.randn(*self.q.shape)
    #     qdot = self.qdot * np.random.randn(*self.q.shape)
    #     qddot = self.qddot * np.random.randn(*self.q.shape)
    #     input = (self.model, q, qdot, qddot)
    #     InverseDynamics(*input)


    # def test_InverseDynamicsGradients(self):

    #     q = self.q * np.random.randn(*self.q.shape)
    #     qdot = self.qdot * np.random.randn(*self.q.shape)
    #     qddot = self.qddot * np.random.randn(*self.q.shape)
    #     input = (self.model["Xtree"], self.model["I"], tuple(self.model["parent"]), tuple(self.model["jtype"]), self.model["jaxis"],
    #              self.model["NB"], q, qdot, qddot, self.model["a_grav"])

    #     InverseDynamicsCore(*input)





if __name__ == "__main__":
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDynamics)
    unittest.TextTestRunner(verbosity=0).run(suite)
