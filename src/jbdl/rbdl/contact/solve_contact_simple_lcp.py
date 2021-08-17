from functools import partial
import numpy as np
from jbdl.rbdl.contact import calc_contact_jacobian_core
from jbdl.rbdl.contact.calc_contact_jacobian import calc_contact_jacobian_core_jit_flag
from jbdl.rbdl.contact.calc_contact_jdot_qdot import calc_contact_jdot_qdot_core
from jbdl.rbdl.contact.calc_contact_jdot_qdot import calc_contact_jdot_qdot_core_jit_flag
# from jbdl.rbdl.contact.SolveContactLCP import quadprog
import jax.numpy as jnp
from jax.api import grad, jit
from jbdl.rbdl.contact import get_contact_force
from jbdl.rbdl.utils import xyz2int


def quad_loss(M, d, lam):
    A = 0.5 * (M + jnp.transpose(M))
    qdloss = 0.5 * jnp.matmul(jnp.transpose(lam), jnp.matmul(A, lam)) + jnp.dot(jnp.transpose(d), lam)
    qdloss = jnp.squeeze(qdloss)
    return qdloss

def non_negative_z_projector(x, nf):
    x = x.at[nf-1::nf].set(jnp.maximum(x[nf-1::nf], 0))
    return x

def solve_contact_simple_lcp_core_jit_flag(x_tree, q, qdot, contactpoint, H, tau, C,  flag_contact, idcontact,  parent, jtype, jaxis, nb, nc, nf):
    Jc = calc_contact_jacobian_core_jit_flag(x_tree, q, contactpoint,flag_contact, idcontact,  parent, jtype, jaxis, nb, nc, nf)
    JcdotQdot = calc_contact_jdot_qdot_core_jit_flag(x_tree, q, qdot, contactpoint, flag_contact, idcontact,  parent, jtype, jaxis, nb, nc, nf)
    tau = jnp.reshape(tau, (-1, 1))
    C = jnp.reshape(C, (-1, 1))
    M = jnp.matmul(Jc, jnp.linalg.solve(H, jnp.transpose(Jc)))
    d0 = jnp.matmul(Jc, jnp.linalg.solve(H, tau - C))
    d = jnp.add(d0, JcdotQdot)
    #Todo: Fast differentiable QP solver.
    lam = -jnp.linalg.solve(M,d)
    lam = non_negative_z_projector(lam, nf)
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
def solve_contact_simple_lcp_core(x_tree, q, qdot, contactpoint, H, tau, C, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf):
    Jc = calc_contact_jacobian_core(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf)
    JcdotQdot = calc_contact_jdot_qdot_core(x_tree, q, qdot, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf)
    tau = jnp.reshape(tau, (-1, 1))
    C = jnp.reshape(C, (-1, 1))
    M = jnp.matmul(Jc, jnp.linalg.solve(H, jnp.transpose(Jc)))
    d0 = jnp.matmul(Jc, jnp.linalg.solve(H, tau - C))
    d = jnp.add(d0, JcdotQdot)
    #Todo: Fast differentiable QP solver.
    lam = -jnp.linalg.solve(M,d)
    lam = non_negative_z_projector(lam, nf)
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

def solve_contact_simple_lcp(model: dict, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray, flag_contact: np.ndarray):
    nc = int(model["nc"])
    nb = int(model["nb"])
    nf = int(model["nf"])
    x_tree = model["x_tree"]
    contactpoint = model["contactpoint"],
    idcontact = tuple(model["idcontact"])
    parent = tuple(model["parent"])
    jtype = tuple(model["jtype"])
    jaxis = xyz2int(model["jaxis"])
    contactpoint = model["contactpoint"]
    flag_contact = flag_contact
    H = model["H"]
    C = model["C"]

    flcp, fqp = solve_contact_simple_lcp_core(x_tree, q, qdot, contactpoint, H, tau, C, \
        idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf)

    fpd = np.zeros((3*nc, 1))
    fc, fcqp, fcpd = get_contact_force(model, fqp, fpd, flag_contact)
    return flcp, fqp, fc, fcqp, fcpd  






