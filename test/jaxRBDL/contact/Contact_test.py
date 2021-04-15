import unittest
import os

from jax.api import jacfwd
from jax.api_util import argnums_partial
from jbdl.rbdl.contact import calc_contact_force_direct, calc_contact_force_direct_core
from jbdl.rbdl.contact import impulsive_dynamics, impulsive_dynamics_core
from jbdl.rbdl.kinematics import calc_point_acceleration_core
from jbdl.rbdl.utils import ModelWrapper
from jbdl.rbdl.contact import calc_contact_jacobian, calc_contact_jacobian_core
from jbdl.rbdl.contact import calc_contact_jdot_qdot, calc_contact_jdot_qdot_core
from jbdl.rbdl.contact import solve_contact_lcp, solve_contact_lcp_core, lcp_quadprog
from jbdl.rbdl.kinematics.calc_point_jacobian import calc_point_jacobian_core
from jbdl.rbdl.dynamics import composite_rigid_body_algorithm
from jbdl.rbdl.dynamics import inverse_dynamics
from jbdl.rbdl.contact import detect_contact, detect_contact_core, determin_contact_type, determin_contact_type_core
from jbdl.rbdl.contact.detect_contact import detect_contact_v0
from jbdl.rbdl.contact import solve_contact_simple_lcp, solve_contact_simple_lcp_core
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
        pass
        # model = self.model
        # idcontact = model["idcontact"]
        # contactpoint = model["contactpoint"]
        # q = self.q  
        # NB = int(model["NB"])
        # NC = int(model["NC"])
        # Xtree = model["Xtree"]
        # parent = tuple(model["parent"])
        # jtype = tuple(model["jtype"])
        # jaxis = model["jaxis"]
        # nf = int(model["nf"])

        # start_time = time.time()
        # for body_id, point_pos in zip(idcontact, contactpoint):
        #     print(body_id, point_pos)
        #     J = calc_point_jacobian_core(Xtree, parent, jtype, jaxis, NB, body_id, q, point_pos)
        #     J.block_until_ready()
        # duration = time.time() - start_time

        # print("Compiled time for calc_point_jacobian_core is %s" % duration)


        # def calc_contact_jacobian_with_jit():
        #     q = self.q * np.random.randn(*self.q.shape)
        #     flag_contact = np.random.randint(3, size=(4,))
        #     input =  (Xtree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
        #     calc_contact_jacobian_core(*input)

        # print("calc_contact_jacobian_core:")
        # print(timeit.Timer(calc_contact_jacobian_with_jit).repeat(repeat=3, number=1000))

    def test_calc_contact_jdot_qdot(self):
        pass

        # model = self.model
        # idcontact = model["idcontact"]
        # contactpoint = model["contactpoint"]
        # q = self.q
        # qdot = self.qdot
        # qddot = self.qddot
        # NB = int(model["NB"])
        # NC = int(model["NC"])
        # Xtree = model["Xtree"]
        # parent = tuple(model["parent"])
        # jtype = tuple(model["jtype"])
        # jaxis = model["jaxis"]
        # nf = int(model["nf"])

        # start_time = time.time()
        # for body_id, point_pos in zip(idcontact, contactpoint):
        #     print(body_id, point_pos)
        #     acc = calc_point_acceleration_core(Xtree, parent, jtype, jaxis, body_id, q, qdot, qddot, point_pos)
        #     acc.block_until_ready()
        # duration = time.time() - start_time
        # print("Compiled time for calc_contact_jdot_qdot_core is %s" % duration)


        # def calc_contact_jdot_qdot_core_with_jit():
        #     q = self.q * np.random.randn(*self.q.shape)
        #     qdot = self.qdot * np.random.randn(*self.qdot.shape)
        #     flag_contact = np.random.randint(3, size=(4,))
        #     calc_contact_jdot_qdot_core(Xtree, q, qdot, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
        

        # print("calc_contact_jdot_qdot_core_with_jit:")
        # print(timeit.Timer(calc_contact_jdot_qdot_core_with_jit).repeat(repeat=3, number=1000))

    def test_calc_contact_force_direct(self):
        flag_contact_list = [np.array([1, 1, 1, 1]), np.array([2, 2, 2, 2])]
        flag_contact = np.array([1, 1, 1, 1])
        model = self.model
        q = self.q
        qdot = self.qdot
        tau = self.tau
        NB = int(model["NB"])

        input_CRBA = (model, q)
        start_time = time.time()
        model["H"] = composite_rigid_body_algorithm(*input_CRBA)
        model["C"] = inverse_dynamics(model, q, qdot, np.zeros((NB, 1)))
        print(time.time()-start_time)
        for flag_contact in flag_contact_list:
            input = (model["Xtree"], self.q, self.qdot, model["contactpoint"],
                    model["H"], tau, model["C"], tuple(model["idcontact"]), tuple(flag_contact),
                        tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
                        model["NB"], model["NC"], model["nf"])
            flcp, _ = calc_contact_force_direct_core(*input)
            flcp.block_until_ready()
        print(time.time()-start_time)

        def calc_contact_force_direct_core_with_jit():
            q = self.q * np.random.randn(*self.q.shape)
            qdot = self.qdot * np.random.randn(*self.qdot.shape)
            idx = np.random.randint(len(flag_contact_list))
            flag_contact = flag_contact_list[idx]
            input = (model["Xtree"], q, qdot, model["contactpoint"],
                 model["H"], tau, model["C"], tuple(model["idcontact"]), tuple(flag_contact),
                    tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
                    model["NB"], model["NC"], model["nf"])
            calc_contact_force_direct_core(*input)

        print(timeit.Timer(calc_contact_force_direct_core_with_jit).repeat(repeat=3, number=1))

        def calc_contact_force_direct_with_jit():
            q = self.q * np.random.randn(*self.q.shape)
            qdot = self.qdot * np.random.randn(*self.qdot.shape)
            idx = np.random.randint(len(flag_contact_list))
            flag_contact = flag_contact_list[idx]
            input = (model, q, qdot, tau, flag_contact)
            calc_contact_force_direct(*input)
        
        print(timeit.Timer(calc_contact_force_direct_with_jit).repeat(repeat=3, number=1))

    def test_detect_contact(self):
        model = self.model
        q = self.q
        qdot =  self.qdot
        input = (model, q, qdot)
        for i in range(1000):
            input = (model, q * np.random.randn(*q.shape), qdot * np.random.randn(*qdot.shape))
            flag_contact_v0 = detect_contact_v0(*input)
            flag_contact_v1 = detect_contact(*input)
            self.assertEqual(np.sum(np.abs(np.array(flag_contact_v0)-np.array(flag_contact_v1))), 0.0)


        # def TimeDetectContac():
        #     q = self.q * np.random.randn(*self.q.shape)
        #     qdot = self.q * np.random.randn(*self.qdot.shape)
        #     model = self.model
        #     input = (model, q, qdot)
        #     detect_contact(*input)
        
        # print(timeit.Timer(TimeDetectContac).repeat(repeat=3, number=1000))

        # model = self.model
        # contact_cond = model["contact_cond"]
       

        # input_2 = (model["Xtree"], q, qdot, model["contactpoint"],
        #             tuple(model["idcontact"]), 
        #             tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
        #             model["NC"])

        # pos, vel = detect_contact_core(*input_2)
        # print(pos)
        # print(vel)
        # input_1 = (pos, vel, \
        #     contact_cond["contact_pos_lb"],
        #     contact_cond["contact_vel_lb"],
        #     contact_cond["contact_vel_ub"])
        # print(DeterminContactTypeCore(*input_1))

        # from jax import make_jaxpr
        # print(make_jaxpr(detect_contact_core, static_argnums=(4, 5, 6, 7, 8, 9, 10, 11))(*input_2))

        # end_pos, end_vel = detect_contact_core(*input_2)
        # from functools import partial

        # print(list(map(partial(DeterminContactType, contact_cond=contact_cond), end_pos, end_vel)))



    
    def test_impulsive_dynamics(self):
        model = self.model
        q = self.q
        qdot = self.qdot
        flag_contact = (1, 1, 1, 1)
        input_A = (model, q, qdot, flag_contact)
        input_CRBA = (model, q)
        model["H"] = composite_rigid_body_algorithm(*input_CRBA)

        rankJc = np.sum( [1 for item in flag_contact if item != 0]) * model["nf"]
        # print(rankJc)

        input = (model["Xtree"], self.q, self.qdot, model["contactpoint"],
                    model["H"], tuple(model["idcontact"]), tuple(flag_contact),
                        tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
                        model["NB"], model["NC"], model["nf"], rankJc)

        impulsive_dynamics_core(*input).block_until_ready()
        # print(impulsive_dynamics(*input_A))
        # from jax import make_jaxpr
        # print(make_jaxpr(impulsive_dynamics_core, static_argnums=(5, 6, 7, 8, 9, 10, 11, 12, 13))(*input))

        def impulsive_dynamics_with_jit():
            impulsive_dynamics(*input_A)

        print(timeit.Timer(impulsive_dynamics_with_jit).repeat(repeat=3, number=1000))

    def test_solve_contact_simple_lcp(self):

        model = self.model
        NB = model["NB"]
        q = self.q
        qdot = self.qdot
        tau = self.tau
        flag_contact = (1, 1, 1, 1)
        input_CRBA = (model, q)
        model["H"] = composite_rigid_body_algorithm(*input_CRBA)
        model["C"] = inverse_dynamics(model, q, qdot, np.zeros((NB, 1)))

        input_core = (model["Xtree"], self.q, self.qdot, model["contactpoint"],
                model["H"], tau, model["C"], tuple(model["idcontact"]), tuple(flag_contact),
                    tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
                    model["NB"], model["NC"], model["nf"])


        input = (model, q, qdot, tau, flag_contact)

        def solve_contact_simple_lcp_with_jit():
            solve_contact_simple_lcp(*input)

        print(timeit.Timer(solve_contact_simple_lcp_with_jit).repeat(repeat=3, number=1))

        def solve_contact_simple_lcp_core_with_jit():
            solve_contact_simple_lcp_core(*input_core)

        print(timeit.Timer(solve_contact_simple_lcp_core_with_jit).repeat(repeat=3, number=1))

    def test_lcp_quadprog(self):
        pass
        # H = jnp.array([[1.0, -1.0],
        #                [-1.0, 2.0]])
        # f = jnp.array([[-2.0], [-6.0]])
        # L = jnp.array([[1.0, 1.0],
        #                [-1.0, 2.0], 
        #                [2.0, 1.0]])
        # k = jnp.array([[2.0], [2.0], [3.0]])

        # lb = jnp.array([[0.0], [0.0]])
        # ub = jnp.array([[0.5], [5.0]])

        # from jax.test_util import check_jvp
        # from jax import jvp
        # from functools import partial
        # check_jvp(lcp_quadprog, partial(jvp, lcp_quadprog), (H, f, L, k, lb, ub))
        # dx2dH = jacfwd(lcp_quadprog, argnums=0)(H, f, L, k, lb, ub)
        # print(dx2dH)
        # print(dx2dH.shape)


    def test_solve_contact_lcp_core_grad(self):
        pass
        # model = self.model
        # NB = model["NB"]
        # NC = model["NC"]
        # q = self.q
        # qdot = self.qdot
        # tau = self.tau
        # flag_contact = (1, 1, 1, 1)
        # input_CRBA = (model, q)
        # model["H"] = composite_rigid_body_algorithm(*input_CRBA)
        # model["C"] = inverse_dynamics(model, q, qdot, np.zeros((NB, 1)))
        # mu = 0.9
        # ncp = 0
        # for i in range(NC):
        #     if flag_contact[i]!=0:
        #         ncp = ncp + 1

        # contact_cond = model["contact_cond"]
        # contact_force_lb = contact_cond["contact_force_lb"].flatten()
        # contact_force_ub = contact_cond["contact_force_ub"].flatten()

    
        # input_core = (model["Xtree"], self.q, self.qdot, model["contactpoint"],
        #         model["H"], tau, model["C"], contact_force_lb, contact_force_ub, tuple(model["idcontact"]), tuple(flag_contact),
        #             tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
        #             model["NB"], model["NC"], model["nf"], ncp, mu)

        # solve_contact_lcp_core(*input_core)

        # def get_lcp(Xtree, q, qdot, contactpoint, H, tau, C, contact_force_lb, contact_force_ub, \
        #     idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, ncp, mu):
        #     flcp, _ = solve_contact_lcp_core(Xtree, q, qdot, contactpoint, H, tau, C, contact_force_lb, contact_force_ub, \
        #     idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, ncp, mu)
        #     return flcp
        # start_time = time.time()
        # get_lcp(*input_core)
        # duration = time.time() - start_time
        # print(duration)

        # from jax.api import jit
        # fun = jacfwd(get_lcp, argnums=1)
        # fun(*input_core)
        # start_time = time.time()
        # input_core = (model["Xtree"], self.q * np.random.randn(*self.q.shape), self.qdot * np.random.randn(*self.qdot.shape), model["contactpoint"],
        #         model["H"], tau, model["C"], contact_force_lb, contact_force_ub, tuple(model["idcontact"]), tuple(flag_contact),
        #             tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
        #             model["NB"], model["NC"], model["nf"], ncp, mu)
        # fun(*input_core)
        # duration = time.time() - start_time
        # print(duration)





    def test_solve_contact_lcp(self):
        pass
        # model = self.model
        # NB = model["NB"]
        # NC = model["NC"]
        # q = self.q
        # qdot = self.qdot
        # tau = self.tau
        # flag_contact = (1, 1, 1, 1)
        # input_CRBA = (model, q)
        # model["H"] = composite_rigid_body_algorithm(*input_CRBA)
        # model["C"] = inverse_dynamics(model, q, qdot, np.zeros((NB, 1)))
        # mu = 0.9
        # ncp = 0
        # for i in range(NC):
        #     if flag_contact[i]!=0:
        #         ncp = ncp + 1

        # contact_cond = model["contact_cond"]
        # contact_force_lb = contact_cond["contact_force_lb"].flatten()
        # contact_force_ub = contact_cond["contact_force_ub"].flatten()


        # input_core = (model["Xtree"], self.q, self.qdot, model["contactpoint"],
        #         model["H"], tau, model["C"], contact_force_lb, contact_force_ub, tuple(model["idcontact"]), tuple(flag_contact),
        #             tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
        #             model["NB"], model["NC"], model["nf"], ncp, mu)

        # solve_contact_lcp_core(*input_core)

        # def solve_contact_lcp_core_with_jit():
        #     q = self.q * np.random.randn(*self.q.shape)
        #     qdot = self.qdot * np.random.randn(*self.qdot.shape)
        #     input_core = (model["Xtree"], q, qdot, model["contactpoint"],
        #         model["H"], tau, model["C"], contact_force_lb, contact_force_ub, tuple(model["idcontact"]), tuple(flag_contact),
        #             tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
        #             model["NB"], model["NC"], model["nf"], ncp, mu)
        #     solve_contact_lcp_core(*input_core)
        
        # print(timeit.Timer(solve_contact_lcp_core_with_jit).repeat(repeat=3, number=100))

        
            






        

        

















        


        



if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestContact)
    unittest.TextTestRunner(verbosity=0).run(suite)
