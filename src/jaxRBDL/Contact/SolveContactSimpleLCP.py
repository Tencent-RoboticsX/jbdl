from functools import partial
import numpy as np
from jaxRBDL.Contact import calc_contact_jacobian_core
from jaxRBDL.Contact.calc_contact_jacobian import calc_contact_jacobian_core_jit_flag
from jaxRBDL.Contact.CalcContactJdotQdot import CalcContactJdotQdotCore, CalcContactJdotQdotCoreJitFlag
from jaxRBDL.Contact.SolveContactLCP import quadprog
import jax.numpy as jnp
from jax.api import grad, jit
from jaxRBDL.Contact.GetContactForce import GetContactForce


def QuadLoss(M, d, lam):
    A = 0.5 * (M + jnp.transpose(M))
    qdloss = 0.5 * jnp.matmul(jnp.transpose(lam), jnp.matmul(A, lam)) + jnp.dot(jnp.transpose(d), lam)
    qdloss = jnp.squeeze(qdloss)
    return qdloss

def NonNegativeZProjector(x, nf):
    x = x.at[nf-1::nf].set(jnp.maximum(x[nf-1::nf], 0))
    return x

def SolveContactSimpleLCPCoreJitFlag(Xtree, q, qdot, contactpoint, H, tau, C,  flag_contact, idcontact,  parent, jtype, jaxis, NB, NC, nf):
    Jc = calc_contact_jacobian_core_jit_flag(Xtree, q, contactpoint,flag_contact, idcontact,  parent, jtype, jaxis, NB, NC, nf)
    JcdotQdot = CalcContactJdotQdotCoreJitFlag(Xtree, q, qdot, contactpoint, flag_contact, idcontact,  parent, jtype, jaxis, NB, NC, nf)
    tau = jnp.reshape(tau, (-1, 1))
    C = jnp.reshape(C, (-1, 1))
    M = jnp.matmul(Jc, jnp.linalg.solve(H, jnp.transpose(Jc)))
    d0 = jnp.matmul(Jc, jnp.linalg.solve(H, tau - C))
    d = jnp.add(d0, JcdotQdot)
    #Todo: Fast differentiable QP solver.
    lam = -jnp.linalg.solve(M,d)
    lam = NonNegativeZProjector(lam, nf)
    # lam = - d / jnp.reshape(jnp.diag(M), (-1, 1))

    # lr = 1e-2
    # for i in range(5):
    #     g = grad(QuadLoss, 2)(M, d, lam)
    #     lam = NonNegativeZProjector(lam - lr * g, nf)
    fqp = lam
    flcp = jnp.matmul(jnp.transpose(Jc), fqp)

    # H = np.asfarray(0.5 * (M+M.transpose()))
    # d = np.asfarray(d)
    # x, status = quadprog(H, d, None, None, None, None, np.zeros_like(lam), None)
    # print(x)
    flcp = jnp.reshape(flcp, (-1,))
    return flcp, fqp

# @partial(jit, static_argnums=(7, 8, 9, 10, 11, 12, 13, 14))
def SolveContactSimpleLCPCore(Xtree, q, qdot, contactpoint, H, tau, C, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf):
    Jc = calc_contact_jacobian_core(Xtree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
    JcdotQdot = CalcContactJdotQdotCore(Xtree, q, qdot, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
    tau = jnp.reshape(tau, (-1, 1))
    C = jnp.reshape(C, (-1, 1))
    M = jnp.matmul(Jc, jnp.linalg.solve(H, jnp.transpose(Jc)))
    d0 = jnp.matmul(Jc, jnp.linalg.solve(H, tau - C))
    d = jnp.add(d0, JcdotQdot)
    #Todo: Fast differentiable QP solver.
    lam = -jnp.linalg.solve(M,d)
    lam = NonNegativeZProjector(lam, nf)
    # lam = - d / jnp.reshape(jnp.diag(M), (-1, 1))

    # lr = 1e-2
    # for i in range(5):
    #     g = grad(QuadLoss, 2)(M, d, lam)
    #     lam = NonNegativeZProjector(lam - lr * g, nf)
    fqp = lam
    flcp = jnp.matmul(jnp.transpose(Jc), fqp)

    # H = np.asfarray(0.5 * (M+M.transpose()))
    # d = np.asfarray(d)
    # x, status = quadprog(H, d, None, None, None, None, np.zeros_like(lam), None)
    # print(x)
    flcp = jnp.reshape(flcp, (-1,))
    return flcp, fqp

def SolveContactSimpleLCP(model: dict, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray, flag_contact: np.ndarray):
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

    flcp, fqp = SolveContactSimpleLCPCore(Xtree, q, qdot, contactpoint, H, tau, C, \
        idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)

    fpd = np.zeros((3*NC, 1))
    fc, fcqp, fcpd = GetContactForce(model, fqp, fpd, flag_contact)
    return flcp, fqp, fc, fcqp, fcpd  






