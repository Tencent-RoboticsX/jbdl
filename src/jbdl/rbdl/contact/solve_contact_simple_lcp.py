import numpy as np
from jbdl.rbdl.contact import calc_contact_jacobian_core
from jbdl.rbdl.contact.calc_contact_jacobian import calc_contact_jacobian_core_jit_flag
from jbdl.rbdl.contact.calc_contact_jdot_qdot import calc_contact_jdot_qdot_core
from jbdl.rbdl.contact.calc_contact_jdot_qdot import calc_contact_jdot_qdot_core_jit_flag
import jax.numpy as jnp
from jbdl.rbdl.contact import get_contact_force
from jbdl.rbdl.utils import xyz2int


def quad_loss(mm, d, lam):
    aa = 0.5 * (mm + jnp.transpose(mm))
    qdloss = 0.5 * jnp.matmul(jnp.transpose(lam), jnp.matmul(aa, lam)) + jnp.dot(jnp.transpose(d), lam)
    qdloss = jnp.squeeze(qdloss)
    return qdloss


def non_negative_z_projector(x, nf):
    x = x.at[nf-1::nf].set(jnp.maximum(x[nf-1::nf], 0))
    return x


def solve_contact_simple_lcp_core_jit_flag(
    x_tree, q, qdot, contactpoint, hh, tau, cc, flag_contact, idcontact,
    parent, jtype, jaxis, nb, nc, nf):

    jc = calc_contact_jacobian_core_jit_flag(
        x_tree, q, contactpoint, flag_contact, idcontact,
        parent, jtype, jaxis, nb, nc, nf)
    jcdot_qdot = calc_contact_jdot_qdot_core_jit_flag(
        x_tree, q, qdot, contactpoint, flag_contact, idcontact,
        parent, jtype, jaxis, nb, nc, nf)

    tau = jnp.reshape(tau, (-1, 1))
    cc = jnp.reshape(cc, (-1, 1))
    mm = jnp.matmul(jc, jnp.linalg.solve(hh, jnp.transpose(jc)))
    d0 = jnp.matmul(jc, jnp.linalg.solve(hh, tau - cc))
    d = jnp.add(d0, jcdot_qdot)
    #Todo: Fast differentiable QP solver.
    lam = -jnp.linalg.solve(mm, d)
    lam = non_negative_z_projector(lam, nf)

    fqp = lam
    flcp = jnp.matmul(jnp.transpose(jc), fqp)

    flcp = jnp.reshape(flcp, (-1,))
    return flcp, fqp


def solve_contact_simple_lcp_core(
    x_tree, q, qdot, contactpoint, hh, tau, cc, idcontact, flag_contact,
    parent, jtype, jaxis, nb, nc, nf):

    jc = calc_contact_jacobian_core(
        x_tree, q, contactpoint, idcontact, flag_contact,
        parent, jtype, jaxis, nb, nc, nf)
    jcdot_qdot = calc_contact_jdot_qdot_core(
        x_tree, q, qdot, contactpoint, idcontact, flag_contact,
        parent, jtype, jaxis, nb, nc, nf)

    tau = jnp.reshape(tau, (-1, 1))
    cc = jnp.reshape(cc, (-1, 1))
    mm = jnp.matmul(jc, jnp.linalg.solve(hh, jnp.transpose(jc)))
    d0 = jnp.matmul(jc, jnp.linalg.solve(hh, tau - cc))
    d = jnp.add(d0, jcdot_qdot)
    #Todo: Fast differentiable QP solver.
    lam = -jnp.linalg.solve(mm, d)
    lam = non_negative_z_projector(lam, nf)
 
    fqp = lam
    flcp = jnp.matmul(jnp.transpose(jc), fqp)

    flcp = jnp.reshape(flcp, (-1,))
    return flcp, fqp

def solve_contact_simple_lcp(model: dict, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray, flag_contact: np.ndarray):
    nc = int(model["nc"])
    nb = int(model["nb"])
    nf = int(model["nf"])
    x_tree = model["x_tree"]
    contactpoint = model["contactpoint"]
    idcontact = tuple(model["idcontact"])
    parent = tuple(model["parent"])
    jtype = tuple(model["jtype"])
    jaxis = xyz2int(model["jaxis"])
    contactpoint = model["contactpoint"]
    flag_contact = flag_contact
    hh = model["H"]
    cc = model["C"]

    flcp, fqp = solve_contact_simple_lcp_core(x_tree, q, qdot, contactpoint, hh, tau, cc, \
        idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf)

    fpd = np.zeros((3*nc, 1))
    fc, fcqp, fcpd = get_contact_force(model, fqp, fpd, flag_contact)
    return flcp, fqp, fc, fcqp, fcpd  
