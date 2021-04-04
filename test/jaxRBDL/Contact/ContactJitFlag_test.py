
import unittest
import os
from jaxRBDL.Contact.CalcContactForceDirect import CalcContactForceDirectCore, CalcContactForceDirect
from jaxRBDL.Contact.ImpulsiveDynamics import ImpulsiveDynamics, ImpulsiveDynamicsCore
from jaxRBDL.Contact.SolveContactLCP import SolveContactLCP
from jaxRBDL.Kinematics.CalcPointAcceleraion import CalcPointAccelerationCore
from jaxRBDL.Utils.ModelWrapper import ModelWrapper
from jaxRBDL.Contact.CalcContactJacobian import CalcContactJacobian, CalcContactJacobianCore, CalcContactJacobianCoreJitFlag
from jaxRBDL.Contact.CalcContactJdotQdot import CalcContactJdotQdotCore, CalcContactJdotQdot, CalcContactJdotQdotCoreJitFlag
from jaxRBDL.Kinematics.calc_point_jacobian import calc_point_jacobian_core
from jaxRBDL.Dynamics.CompositeRigidBodyAlgorithm import CompositeRigidBodyAlgorithm
from jaxRBDL.Dynamics.InverseDynamics import InverseDynamics
from jaxRBDL.Contact.DetectContact import DetectContact_v0, DetectContact, DetectContactCore, DeterminContactType, DeterminContactTypeCore
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

    def test_CalcContactJacobianCore(self):


        pass
        
        model = self.model
        nf = model["nf"]
        

        idcontact = model["idcontact"]
        contactpoint = model["contactpoint"]
        xs = (idcontact, contactpoint)
        # from jax.tree_util import tree_structure
        # print(tree_structure(xs))
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


            J = CalcContactJacobianCoreJitFlag(*input)
            J = np.matmul(flag_contact_mat, J)       
            Js = CalcContactJacobianCore(*input)

            self.assertEqual(np.sum(np.abs(J-Js)), 0.0)

    def test_CalcContactJdotQdotCore(self):
        pass
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
            # input_jit_flag = (model["Xtree"], q, qdot, model["contactpoint"], flag_contact,
            #             tuple(model["idcontact"]), 
            #             tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
            #             model["NB"], model["NC"], model["nf"])

            output = CalcContactJdotQdotCore(*input)
            output_jit_flag = CalcContactJdotQdotCoreJitFlag(*input)
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