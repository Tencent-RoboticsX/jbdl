import unittest
import os
from jaxRBDL.Contact.CalcContactForceDirect import CalcContactForceDirectCore, CalcContactForceDirect
from jaxRBDL.Contact.ImpulsiveDynamics import ImpulsiveDynamics, ImpulsiveDynamicsCore
from jaxRBDL.Contact.SolveContactLCP import SolveContactLCP
from jaxRBDL.Kinematics import calc_point_acceleration_core
from jaxRBDL.Utils.ModelWrapper import ModelWrapper
from jaxRBDL.Contact import calc_contact_jacobian, calc_contact_jacobian_core
from jaxRBDL.Contact import calc_contact_jdot_qdot, calc_contact_jdot_qdot_core
from jaxRBDL.Kinematics.calc_point_jacobian import calc_point_jacobian_core
from jaxRBDL.Dynamics import composite_rigid_body_algorithm
from jaxRBDL.Dynamics.InverseDynamics import InverseDynamics
from jaxRBDL.Contact.DetectContact import DetectContact_v0, DetectContact, DetectContactCore, DeterminContactType, DeterminContactTypeCore
from jaxRBDL.Contact.SolveContactSimpleLCP import QuadLoss, NonNegativeZProjector, SolveContactSimpleLCPCore, SolveContactSimpleLCP
import numpy as np
from test.support import EnvironmentVarGuard
import time
import timeit
import timeit, functools
import jax.numpy as jnp

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

    def test_calc_contact_jacobian(self):
        model = self.model
        idcontact = model["idcontact"]
        contactpoint = model["contactpoint"]
        q = self.q  
        NB = int(model["NB"])
        NC = int(model["NC"])
        Xtree = model["Xtree"]
        parent = tuple(model["parent"])
        jtype = tuple(model["jtype"])
        jaxis = model["jaxis"]
        nf = int(model["nf"])

        start_time = time.time()
        for body_id, point_pos in zip(idcontact, contactpoint):
            print(body_id, point_pos)
            J = calc_point_jacobian_core(Xtree, parent, jtype, jaxis, NB, body_id, q, point_pos)
            J.block_until_ready()
        duration = time.time() - start_time

        print("Compiled time for calc_point_jacobian_core is %s" % duration)


        def calc_contact_jacobian_with_jit():
            q = self.q * np.random.randn(*self.q.shape)
            flag_contact = np.random.randint(3, size=(4,))
            input =  (Xtree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
            calc_contact_jacobian_core(*input)

        print("calc_contact_jacobian_core:")
        print(timeit.Timer(calc_contact_jacobian_with_jit).repeat(repeat=3, number=1000))

    def test_calc_contact_jdot_qdot(self):

        model = self.model
        idcontact = model["idcontact"]
        contactpoint = model["contactpoint"]
        q = self.q
        qdot = self.qdot
        qddot = self.qddot
        NB = int(model["NB"])
        NC = int(model["NC"])
        Xtree = model["Xtree"]
        parent = tuple(model["parent"])
        jtype = tuple(model["jtype"])
        jaxis = model["jaxis"]
        nf = int(model["nf"])

        start_time = time.time()
        for body_id, point_pos in zip(idcontact, contactpoint):
            print(body_id, point_pos)
            acc = calc_point_acceleration_core(Xtree, parent, jtype, jaxis, body_id, q, qdot, qddot, point_pos)
            acc.block_until_ready()
        duration = time.time() - start_time
        print("Compiled time for calc_contact_jdot_qdot_core is %s" % duration)


        def calc_contact_jdot_qdot_core_with_jit():
            q = self.q * np.random.randn(*self.q.shape)
            qdot = self.qdot * np.random.randn(*self.qdot.shape)
            flag_contact = np.random.randint(3, size=(4,))
            calc_contact_jdot_qdot_core(Xtree, q, qdot, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
        

        print("calc_contact_jdot_qdot_core_with_jit:")
        print(timeit.Timer(calc_contact_jdot_qdot_core_with_jit).repeat(repeat=3, number=1000))

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
        # model["H"] = composite_rigid_body_algorithm(*input_CRBA)
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
        # qdot =  self.qdot
        # input = (model, q, qdot)
        # for i in range(1000):
        #     input = (model, q * np.random.randn(*q.shape), qdot * np.random.randn(*qdot.shape))
        #     flag_contact_v0 = DetectContact_v0(*input)
        #     flag_contact_v1 = DetectContact(*input)
        #     self.assertEqual(np.sum(np.abs(np.array(flag_contact_v0)-np.array(flag_contact_v1))), 0.0)


        # def TimeDetectContac():
        #     q = self.q * np.random.randn(*self.q.shape)
        #     qdot = self.q * np.random.randn(*self.qdot.shape)
        #     model = self.model
        #     input = (model, q, qdot)
        #     DetectContact(*input)
        
        # print(timeit.Timer(TimeDetectContac).repeat(repeat=3, number=1000))

        # model = self.model
        # contact_cond = model["contact_cond"]
       

        # input_2 = (model["Xtree"], q, qdot, model["contactpoint"],
        #             tuple(model["idcontact"]), 
        #             tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
        #             model["NC"])

        # pos, vel = DetectContactCore(*input_2)
        # print(pos)
        # print(vel)
        # input_1 = (pos, vel, \
        #     contact_cond["contact_pos_lb"],
        #     contact_cond["contact_vel_lb"],
        #     contact_cond["contact_vel_ub"])
        # print(DeterminContactTypeCore(*input_1))

        # from jax import make_jaxpr
        # print(make_jaxpr(DetectContactCore, static_argnums=(4, 5, 6, 7, 8, 9, 10, 11))(*input_2))

        # end_pos, end_vel = DetectContactCore(*input_2)
        # from functools import partial

        # print(list(map(partial(DeterminContactType, contact_cond=contact_cond), end_pos, end_vel)))



    
    def test_ImpulsiveDynamics(self):
        pass
        # model = self.model
        # q = self.q
        # qdot = self.qdot
        # flag_contact = (1, 1, 1, 1)
        # input_A = (model, q, qdot, flag_contact)
        # input_CRBA = (model, q)
        # model["H"] = composite_rigid_body_algorithm(*input_CRBA)

        # rankJc = np.sum( [1 for item in flag_contact if item != 0]) * model["nf"]
        # # print(rankJc)

        # input = (model["Xtree"], self.q, self.qdot, model["contactpoint"],
        #             model["H"], tuple(model["idcontact"]), tuple(flag_contact),
        #                 tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
        #                 model["NB"], model["NC"], model["nf"], rankJc)

        # ImpulsiveDynamicsCore(*input).block_until_ready()
        # # print(ImpulsiveDynamics(*input_A))
        # # from jax import make_jaxpr
        # # print(make_jaxpr(ImpulsiveDynamicsCore, static_argnums=(5, 6, 7, 8, 9, 10, 11, 12, 13))(*input))

        # def ImpulsiveDynamicsWithJit():
        #     ImpulsiveDynamics(*input_A)

        # print(timeit.Timer(ImpulsiveDynamicsWithJit).repeat(repeat=3, number=1000))

    def test_SolveContactSimpleLCP(self):
        pass
        # model = self.model
        # NB = model["NB"]
        # q = self.q
        # qdot = self.qdot
        # tau = self.tau
        # flag_contact = (1, 1, 1, 1)
        # input_CRBA = (model, q)
        # model["H"] = composite_rigid_body_algorithm(*input_CRBA)
        # model["C"] = InverseDynamics(model, q, qdot, np.zeros((NB, 1)))

        # input_core = (model["Xtree"], self.q, self.qdot, model["contactpoint"],
        #         model["H"], tau, model["C"], tuple(model["idcontact"]), tuple(flag_contact),
        #             tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
        #             model["NB"], model["NC"], model["nf"])


        # input = (model, q, qdot, tau, flag_contact)

        # def CalContactSimpleLCPWithJit():
        #     SolveContactSimpleLCP(*input)

        # print(timeit.Timer(CalContactSimpleLCPWithJit).repeat(repeat=3, number=1))

        # def CalContactSimpleLCPCoreWithJit():
        #     SolveContactSimpleLCPCore(*input_core)

        # print(timeit.Timer(CalContactSimpleLCPCoreWithJit).repeat(repeat=3, number=1))


        

        

















        


        



if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestContact)
    unittest.TextTestRunner(verbosity=0).run(suite)
