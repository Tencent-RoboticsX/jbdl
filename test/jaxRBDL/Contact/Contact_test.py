import unittest
import os
from jaxRBDL.Contact.CalcContactForceDirect import CalcContactForceDirectCore, CalcContactForceDirect
from jaxRBDL.Contact.ImpulsiveDynamics import ImpulsiveDynamics, ImpulsiveDynamicsCore
from jaxRBDL.Kinematics.CalcPointAcceleraion import CalcPointAccelerationCore
from jaxRBDL.Utils.ModelWrapper import ModelWrapper
from jaxRBDL.Contact.CalcContactJacobian import CalcContactJacobian, CalcContactJacobianCore
from jaxRBDL.Contact.CalcContactJdotQdot import CalcContactJdotQdotCore, CalcContactJdotQdot
from jaxRBDL.Kinematics.CalcPointJacobian import CalcPointJacobianCore
from jaxRBDL.Dynamics.CompositeRigidBodyAlgorithm import CompositeRigidBodyAlgorithm
from jaxRBDL.Dynamics.InverseDynamics import InverseDynamics
from jaxRBDL.Contact.DetectContact import DetectContact, DetectContactCore, DeterminContactType
import numpy as np
from test.support import EnvironmentVarGuard
import time
import timeit
import timeit, functools

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(CURRENT_PATH), "Data")
print(DATA_PATH)


class TestContact(unittest.TestCase):
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
        self.env32cpu = EnvironmentVarGuard()
        self.env32cpu.set('JAX_PLATFORM_NAME', 'cpu')  
        self.env64cpu = EnvironmentVarGuard()
        self.env64cpu.set('JAX_ENABLE_X64', '1')
        self.env64cpu.set('JAX_PLATFORM_NAME', 'cpu')  
        self.env64cpu_nojit = EnvironmentVarGuard()
        self.env64cpu_nojit.set('JAX_ENABLE_X64', '1')
        self.env64cpu_nojit.set('JAX_PLATFORM_NAME', 'cpu')  
        self.env64cpu_nojit.set('JAX_DISABLE_JIT', '1')
        self.startTime = time.time()

    def test_CalcContactJacobian(self):
        pass
        # flag_contact_list = [np.array([1, 1, 1, 1]), np.array([2, 2, 2, 2])]
        # model = self.model
        # for flag_contact in flag_contact_list:
        #     input = (model["Xtree"], self.q, model["contactpoint"],
        #             tuple(model["idcontact"]), tuple(flag_contact),
        #             tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
        #             model["NB"], model["NC"], model["nf"])
        #     CalcContactJacobianCore(*input).block_until_ready()
        
        # # from jax import make_jaxpr
        # # print(make_jaxpr(CalcContactJacobianCore, static_argnums=(3, 4, 5, 6, 7, 8, 9, 10))(*input))
        # # print(CalcContactJacobianCore(*input))
        # def CalcContactJacobianWithJit():
        #     q = self.q * np.random.randn(*self.q.shape)
        #     idx = np.random.randint(len(flag_contact_list))
        #     contact_flag = flag_contact_list[idx]
        #     input =  (self.model, q, contact_flag)
        #     CalcContactJacobian(*input)

        # print("CalcContactJacobian:")
        # print(timeit.Timer(CalcContactJacobianWithJit).repeat(repeat=3, number=1000))

    def test_CalcContactJdotQdot(self):
        pass
        # flag_contact_list = [np.array([1, 1, 1, 1]), np.array([2, 2, 2, 2])]
        # for flag_contact in flag_contact_list:
        #     model = self.model
        #     input = (model["Xtree"], self.q, self.qdot, model["contactpoint"],
        #             tuple(model["idcontact"]), tuple(flag_contact),
        #             tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
        #             model["NB"], model["NC"], model["nf"])
        #     CalcContactJdotQdotCore(*input).block_until_ready()
        # # from jax import make_jaxpr
        # # print(make_jaxpr(CalcContactJdotQdotCore, static_argnums=(5, 6, 7, 8, 9, 10, 11, 12))(*input))

        # def CalcContactJdotQdotCoreWithJit():
        #     q = self.q * np.random.randn(*self.q.shape)
        #     qdot = self.qdot * np.random.randn(*self.qdot.shape)
        #     idx = np.random.randint(len(flag_contact_list))
        #     contact_flag = flag_contact_list[idx]
        #     input =  (self.model, q, qdot, contact_flag)
        #     CalcContactJdotQdot(*input)

        # print("CalcContactJdotQdotCoreWithJit:")
        # print(timeit.Timer(CalcContactJdotQdotCoreWithJit).repeat(repeat=3, number=1000))

    def test_CalcContactForceDirect(self):
        pass
        # flag_contact_list = [np.array([1, 1, 1, 1]), np.array([2, 2, 2, 2])]
        # flag_contact = np.array([1, 1, 1, 1])
        # model = self.model
        # q = self.q
        # qdot = self.qdot
        # tau = self.tau
        # NB = int(model["NB"])

        # input_CRBA = (model, q)
        # start_time = time.time()
        # model["H"] = CompositeRigidBodyAlgorithm(*input_CRBA)
        # model["C"] = InverseDynamics(model, q, qdot, np.zeros((NB, 1)))
        # print(time.time()-start_time)
        # for flag_contact in flag_contact_list:
        #     input = (model["Xtree"], self.q, self.qdot, model["contactpoint"],
        #             model["H"], tau, model["C"], tuple(model["idcontact"]), tuple(flag_contact),
        #                 tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
        #                 model["NB"], model["NC"], model["nf"])
        #     _, flcp = CalcContactForceDirectCore(*input)
        #     flcp.block_until_ready()
        # print(time.time()-start_time)

        # def CalcContactForceDirectCoreWithJit():
        #     q = self.q * np.random.randn(*self.q.shape)
        #     qdot = self.qdot * np.random.randn(*self.qdot.shape)
        #     idx = np.random.randint(len(flag_contact_list))
        #     flag_contact = flag_contact_list[idx]
        #     input = (model["Xtree"], q, qdot, model["contactpoint"],
        #          model["H"], tau, model["C"], tuple(model["idcontact"]), tuple(flag_contact),
        #             tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
        #             model["NB"], model["NC"], model["nf"])
        #     CalcContactForceDirectCore(*input)

        # print(timeit.Timer(CalcContactForceDirectCoreWithJit).repeat(repeat=3, number=1))

        # def CalcContactForceDirectWithJit():
        #     q = self.q * np.random.randn(*self.q.shape)
        #     qdot = self.qdot * np.random.randn(*self.qdot.shape)
        #     idx = np.random.randint(len(flag_contact_list))
        #     flag_contact = flag_contact_list[idx]
        #     input = (model, q, qdot, tau, flag_contact)
        #     CalcContactForceDirect(*input)
        
        # print(timeit.Timer(CalcContactForceDirectWithJit).repeat(repeat=3, number=1))

    def test_DetectContact(self):
        pass
        # model = self.model
        # q = self.q
        # qdot = self.qdot
        # input = (model, q, qdot)
        # DetectContact(*input)

        # def TimeDetectContac():
        #     q = self.q * np.random.randn(*self.q.shape)
        #     qdot = self.q * np.random.randn(*self.qdot.shape)
        #     model = self.model
        #     input = (model, q, qdot)
        #     DetectContact(*input)
        
        # print(timeit.Timer(TimeDetectContac).repeat(repeat=3, number=1000))

        # model = self.model
        # contact_cond = model["contact_cond"]
        # input_1 = (pos, vel, contact_pos_lb, contact_vel_lb, contact_vel_ub)

        # input_2 = (model["Xtree"], self.q, self.qdot, model["contactpoint"],
        #             tuple(model["idcontact"]), 
        #             tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
        #             model["NC"])
        # from jax import make_jaxpr
        # print(make_jaxpr(DetectContactCore, static_argnums=(4, 5, 6, 7, 8, 9, 10, 11))(*input_2))

        # end_pos, end_vel = DetectContactCore(*input_2)
        # from functools import partial

        # print(list(map(partial(DeterminContactType, contact_cond=contact_cond), end_pos, end_vel)))


        # DetectContactCore(*input_2)
    
    def test_ImpulsiveDynamics(self):
        model = self.model
        q = self.q
        qdot = self.qdot
        flag_contact = (1, 1, 1, 1)
        input_A = (model, q, qdot, flag_contact)
        input_CRBA = (model, q)
        model["H"] = CompositeRigidBodyAlgorithm(*input_CRBA)

        rankJc = np.sum( [1 for item in flag_contact if item != 0]) * model["nf"]
        # print(rankJc)

        input = (model["Xtree"], self.q, self.qdot, model["contactpoint"],
                    model["H"], tuple(model["idcontact"]), tuple(flag_contact),
                        tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
                        model["NB"], model["NC"], model["nf"], rankJc)

        ImpulsiveDynamicsCore(*input).block_until_ready()
        # print(ImpulsiveDynamics(*input_A))
        # from jax import make_jaxpr
        # print(make_jaxpr(ImpulsiveDynamicsCore, static_argnums=(5, 6, 7, 8, 9, 10, 11, 12, 13))(*input))

        def ImpulsiveDynamicsWithJit():
            ImpulsiveDynamics(*input_A)

        print(timeit.Timer(ImpulsiveDynamicsWithJit).repeat(repeat=3, number=1000))















        


        



if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestContact)
    unittest.TextTestRunner(verbosity=0).run(suite)
