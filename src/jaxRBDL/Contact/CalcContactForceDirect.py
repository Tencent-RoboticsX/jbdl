from functools import partial
from operator import matmul
from typing import Tuple
import numpy as np
from numpy import linalg
from jaxRBDL.Contact import calc_contact_jacobian, calc_contact_jacobian_core
from jaxRBDL.Contact import calc_contact_jdot_qdot, calc_contact_jdot_qdot_core
from jaxRBDL.Contact.CalcContactForcePD import CalcContactForcePD
from jaxRBDL.Contact import get_contact_force
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import gmres
from jax.api import jit
import jax.numpy as jnp

def CheckContactForce(model: dict, flag_contact: np.ndarray, fqp: np.ndarray):
    NC = int(model["NC"])
    nf = int(model["nf"])
    flag_contact = flag_contact

    flag_recalc = 0
    flag_newcontact = list(flag_contact)

    k = 0
    for i in range(NC):
        if flag_contact[i] != 0:
            if fqp[k*nf+nf-1, 0] < 0:
                flag_newcontact[i] = 0
                flag_recalc = 1
                break
            k = k+1
    flag_newcontact = tuple(flag_newcontact)
    return flag_newcontact, flag_recalc

# @partial(jit, static_argnums=(7, 8, 9, 10, 11, 12, 13, 14))
def CalcContactForceDirectCore(Xtree, q, qdot, contactpoint, H, tau, C, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf):
    Jc = calc_contact_jacobian_core(Xtree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
    JcdotQdot = calc_contact_jdot_qdot_core(Xtree, q, qdot, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
    tau = jnp.reshape(tau, (-1, 1))
    C = jnp.reshape(C, (-1, 1))
    M = jnp.matmul(Jc, jnp.linalg.solve(H, jnp.transpose(Jc)))
    d0 = jnp.matmul(Jc, jnp.linalg.solve(H, tau - C))
    # M = jnp.matmul(jnp.matmul(Jc, Hinv), jnp.transpose(Jc))
    # d0 = jnp.matmul(np.matmul(Jc, Hinv), tau - C)
    d = jnp.add(d0, JcdotQdot)
    fqp = -jnp.linalg.solve(M,d)
    flcp = jnp.matmul(jnp.transpose(Jc), fqp)
    flcp = jnp.reshape(flcp, (-1,))
    return flcp, fqp
        
def CalcContactForceDirect(model: dict, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray, flag_contact: np.ndarray):
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
    flag_contact = flag_contact
    H = model["H"]
    C = model["C"]
        
    flag_recalc = 1
    fqp = np.empty((0, 1))
    flcp = np.empty((0, 1))
    print("++++++++++++++++++++")
    while flag_recalc:
        if np.sum(flag_contact)==0:
            fqp = np.zeros((NC*nf, 1))
            flcp = np.zeros((NB, 1))
            fc = np.zeros((3*NC,))
            fcqp = np.zeros((3*NC,))
            fcpd = np.zeros((3*NC,))
            break

        print("-------------------")
        print(flag_contact)
        flcp, fqp = CalcContactForceDirectCore(Xtree, q, qdot, contactpoint, H, tau, C, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
        print(flag_contact)

        # Check whether the Fz is positive

        flag_contact, flag_recalc = CheckContactForce(model, flag_contact, fqp)
        
    # Calculate contact force from PD controller
    # fpd = CalcContactForcePD(model, q, qdot, flag_contact)
    # fc, fcqp, fcpd = get_contact_force(model, fqp, fpd, flag_contact)  
    fpd = np.zeros((3*NC, 1))
    fc, fcqp, fcpd = get_contact_force(model, fqp, fpd, flag_contact)  

    print("=======================")

    return flcp, fqp, fc, fcqp, fcpd


# def CalcContactForceDirect(model: dict, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray, flag_contact: np.ndarray):
#     NB = int(model["NB"])
#     NC = int(model["NC"])
#     nf = int(model["nf"])
        
#     flag_recalc = 1
#     fqp = np.empty((0, 1))
#     flcp = np.empty((0, 1))  
#     while flag_recalc:
#         if np.sum(flag_contact)==0:
#             fqp = np.zeros((NC*nf, 1))
#             flcp = np.zeros((NB, 1))
#             fc = np.zeros((3*NC,))
#             fcqp = np.zeros((3*NC,))
#             fcpd = np.zeros((3*NC,))
#             break


#         # Calculate contact force
#         Jc = calc_contact_jacobian(model, q, flag_contact)
#         JcdotQdot = calc_contact_jdot_qdot(model, q, qdot, flag_contact)

#         M = np.matmul(np.matmul(Jc, model["Hinv"]), np.transpose(Jc))
#         # print(M)
#         tau = tau.reshape(-1, 1)
#         d0 = np.matmul(np.matmul(Jc, model["Hinv"]), tau - model["C"])
#         d = np.add(d0, JcdotQdot )
        
#         #TODO M may be sigular for nf=3 
#         fqp = -np.linalg.solve(M,d)
     

#         # Check whether the Fz is positive
#         flag_contact, flag_recalc = CheckContactForce(model, flag_contact, fqp)
#         if flag_recalc == 0:
#             flcp = np.matmul(np.transpose(Jc), fqp)
        
#     # Calculate contact force from PD controller
#     fpd = CalcContactForcePD(model, q, qdot, flag_contact)
#     fc, fcqp, fcpd = get_contact_force(model, fqp, fpd, flag_contact)  


#     return flcp, fqp, fc, fcqp, fcpd