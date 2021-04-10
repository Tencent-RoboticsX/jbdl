
import unittest
import os
from jaxRBDL.Contact.CalcContactForceDirect import CalcContactForceDirectCore, CalcContactForceDirect
from jaxRBDL.Contact import impulsive_dynamics, impulsive_dynamics_core
from jaxRBDL.Kinematics import calc_point_acceleration_core
from jaxRBDL.Utils.ModelWrapper import ModelWrapper
from jaxRBDL.Contact import calc_contact_jacobian, calc_contact_jacobian_core
from jaxRBDL.Contact.calc_contact_jacobian import calc_contact_jacobian_core_jit_flag
from jaxRBDL.Contact import calc_contact_jdot_qdot, calc_contact_jdot_qdot_core
from jaxRBDL.Contact.calc_contact_jdot_qdot import calc_contact_jdot_qdot_core_jit_flag
from jaxRBDL.Kinematics.calc_point_jacobian import calc_point_jacobian_core
from jaxRBDL.Dynamics import composite_rigid_body_algorithm
from jaxRBDL.Dynamics import inverse_dynamics
from jaxRBDL.Contact.detect_contact import detect_contact_v0
from jaxRBDL.Contact import detect_contact, detect_contact_core, determin_contact_type, determin_contact_type_core
from jaxRBDL.Contact.SolveContactSimpleLCP import QuadLoss, NonNegativeZProjector, SolveContactSimpleLCPCore, SolveContactSimpleLCP
import numpy as np
from test.support import EnvironmentVarGuard
import time
import timeit
import timeit, functools
import jax.numpy as jnp
from jax import lax


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(CURRENT_PATH), "Data")


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
    
    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_calc_contact_jacobian_core_jit_flag(self):        
        model = self.model
        nf = model["nf"]
        

        idcontact = model["idcontact"]
        contactpoint = model["contactpoint"]
        xs = (idcontact, contactpoint)
        Xtree = model["Xtree"]
        parent = model["parent"]
        q = self.q
        jtype = model["jtype"]
        jaxis = model["jaxis"]
        NB = model["NB"]

        flag_contact_list = list(np.random.randint(3, size=(10,4)))
        for flag_contact in flag_contact_list:
            print(flag_contact)
            flag_contact_mat = np.diag(np.repeat(np.heaviside(flag_contact, 0.0), nf))
            flag_contact_mat = flag_contact_mat[np.any(flag_contact_mat, axis=0), :]
            q = q * np.random.randn(*q.shape)
            input = (model["Xtree"], q, model["contactpoint"],
                    tuple(model["idcontact"]), flag_contact,
                    tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
                    model["NB"], model["NC"], model["nf"])


            J = calc_contact_jacobian_core_jit_flag(*input)
            J = np.matmul(flag_contact_mat, J)       
            Js = calc_contact_jacobian_core(*input)

            self.assertEqual(np.sum(np.abs(J-Js)), 0.0)

    def test_calc_contact_jdot_qdot_core(self):

        model = self.model
        q = self.q
        qdot = self.qdot
        nf = model["nf"]

        flag_contact_list = list(np.random.randint(3, size=(10,4)))
        for flag_contact in flag_contact_list:

            print(flag_contact)
            flag_contact_mat = np.diag(np.repeat(np.heaviside(flag_contact, 0.0), nf))
            flag_contact_mat = flag_contact_mat[np.any(flag_contact_mat, axis=0), :]
            input = (model["Xtree"], q, qdot, model["contactpoint"],
                        tuple(model["idcontact"]), flag_contact,
                        tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
                        model["NB"], model["NC"], model["nf"])
            output = calc_contact_jdot_qdot_core(*input)
            output_jit_flag = calc_contact_jdot_qdot_core_jit_flag(*input)
            output_jit_flag = np.matmul(flag_contact_mat, output_jit_flag)

            self.assertEqual(np.sum(np.abs(output-output_jit_flag)), 0.0)


    def test_FlagContactMat(self):
        pass

        # def get_flag_contact_mat(flag_contact, nf):
        #     flag_contact_mat = jnp.diag(jnp.repeat(jnp.heaviside(flag_contact, 0.0), nf))
        #     # flag_contact_mat = flag_contact_mat[jnp.any(flag_contact_mat, axis=0), :]
        #     return flag_contact_mat

        # flag_contact = np.array([0, 0, 1, 1])
        # nf = 3
        # input = (flag_contact, nf)
        # print(get_flag_contact_mat(*input))


        # from jax import make_jaxpr

        # print(make_jaxpr(get_flag_contact_mat, static_argnums=(1,))(*input))



      

    


if __name__ == "__main__":
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestContact)
    unittest.TextTestRunner(verbosity=0).run(suite)