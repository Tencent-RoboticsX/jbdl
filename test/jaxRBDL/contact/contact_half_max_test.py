import os
from jbdl.rbdl.utils import ModelWrapper
import numpy as np
import unittest
import math
import time
import jax.numpy as jnp
import timeit
from jbdl.rbdl.dynamics import composite_rigid_body_algorithm, inverse_dynamics
from jbdl.rbdl.utils import xyz2int
from jbdl.rbdl.contact.solve_contact_lcp import solve_contact_lcp_extend_core 
from jbdl.rbdl.dynamics.state_fun_ode import dynamics_fun_core
from jbdl.rbdl.utils.xyz2int import xyz2int
from functools import partial
from jax.custom_derivatives import closure_convert

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(CURRENT_PATH), "Data")
print(DATA_PATH)

class TestContact(unittest.TestCase):
    def setUp(self):
        mdlw = ModelWrapper()
        mdlw.load(os.path.join(DATA_PATH, 'half_max_v1.json'))
        self.model = mdlw.model
        self.q = np.array([0.0,  0.4125, 0.0, math.pi/6, math.pi/6, -math.pi/3, -math.pi/3])
        self.NB = self.model["NB"]
        self.NC = self.model["NC"]
        self.qdot = np.ones(self.NB)
        self.qddot = np.ones(self.NB)
        self.tau = np.concatenate([np.zeros(3), np.ones(self.NB-3)])
        self.startTime = time.time()

    
    def test_solve_contact_lcp_extend_core(self):
        # pass
        model = self.model
        NB = model["NB"]
        NC = model["NC"]
        q = self.q
        qdot = self.qdot
        x = np.hstack([q, qdot])
        tau = self.tau
        flag_contact = jnp.array((1.0, 1.0))
        t = 0.0
        
        input_CRBA = (model, q)
        model["H"] = composite_rigid_body_algorithm(*input_CRBA)
        model["C"] = inverse_dynamics(model, q, qdot, np.zeros((NB, 1)))
        mu = 0.9
        ncp = 0
        for i in range(NC):
            if flag_contact[i]!=0:
                ncp = ncp + 1

        contact_cond = model["contact_cond"]
        contact_force_lb = contact_cond["contact_force_lb"].flatten()
        contact_force_ub = contact_cond["contact_force_ub"].flatten()


        input_core = (model["Xtree"], self.q, self.qdot, model["contactpoint"],
                model["H"], tau, model["C"], contact_force_lb, contact_force_ub,
                tuple(model["idcontact"]), flag_contact,
                tuple(model["parent"]), tuple(model["jtype"]), xyz2int(model["jaxis"]),
                model["NB"], model["NC"], model["nf"], ncp, mu)

        print(solve_contact_lcp_extend_core(*input_core))

        # def solve_contact_lcp_core_extend_with_jit():
        #     q = self.q * np.random.randn(*self.q.shape)
        #     qdot = self.qdot * np.random.randn(*self.qdot.shape)
        #     input_core = (model["Xtree"], q, qdot, model["contactpoint"],
        #         model["H"], tau, model["C"], contact_force_lb, contact_force_ub, tuple(model["idcontact"]), flag_contact,
        #             tuple(model["parent"]), tuple(model["jtype"]), xyz2int(model["jaxis"]),
        #             model["NB"], model["NC"], model["nf"], ncp, mu)
        #     solve_contact_lcp_extend_core(*input_core)

        # print(timeit.Timer(solve_contact_lcp_core_extend_with_jit).repeat(repeat=3, number=100))

        def dynamics_fun(x, t, Xtree, I, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub, mu,\
            flag_contact, idcontact,  parent, jtype, jaxis, NB, NC, nf, ncp):
            q = x[0:NB]
            qdot = x[NB:]
            xdot,fqp, H = dynamics_fun_core(Xtree, I, q, qdot, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub,\
                idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, ncp, mu)
            return xdot

        Xtree = model["Xtree"]
        I = model["I"]
        contactpoint = model["contactpoint"]
        a_grav = model["a_grav"]
        idcontact = tuple(model["idcontact"])
        parent = tuple(model["parent"])
        jtype = tuple(model["jtype"])
        jaxis = xyz2int(model["jaxis"])
        NB = model["NB"]
        NC = model["NC"]
        nf = model["nf"]
        


        xdot = dynamics_fun(x, t, Xtree, I, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub, mu, \
            flag_contact, idcontact, parent, jtype, jaxis, NB, NC, nf, ncp)
        print(xdot)

        pure_dynamics_fun = partial(dynamics_fun, idcontact=idcontact, parent=parent, jtype=jtype, jaxis=jaxis, NB=NB, NC=NC, nf=nf, ncp=ncp)
        pure_args = (x, t, Xtree, I, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub, mu, flag_contact)

        converted, consts = closure_convert(pure_dynamics_fun, *pure_args)



    
if __name__ == "__main__":
    # print("=====")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestContact)
    unittest.TextTestRunner(verbosity=0).run(suite)
