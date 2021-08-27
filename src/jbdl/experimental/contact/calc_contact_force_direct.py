import numpy as np
from jbdl.rbdl.contact import calc_contact_jacobian_core
from jbdl.rbdl.contact import calc_contact_jdot_qdot_core
from jbdl.rbdl.contact import get_contact_force
import jax.numpy as jnp
from jbdl.rbdl.utils import xyz2int


def check_contact_force(model: dict, flag_contact: np.ndarray, fqp: np.ndarray):
    nc = int(model["nc"])
    nf = int(model["nf"])
    flag_contact = flag_contact

    flag_recalc = 0
    flag_newcontact = list(flag_contact)

    k = 0
    for i in range(nc):
        if flag_contact[i] != 0:
            if fqp[k*nf+nf-1, 0] < 0:
                flag_newcontact[i] = 0
                flag_recalc = 1
                break
            k = k+1
    flag_newcontact = tuple(flag_newcontact)
    return flag_newcontact, flag_recalc


# @partial(jit, static_argnums=(7, 8, 9, 10, 11, 12, 13, 14))
def calc_contact_force_direct_core(
    x_tree, q, qdot, contactpoint, h, tau, c, idcontact, flag_contact,
    parent, jtype, jaxis, nb, nc, nf):

    jc = calc_contact_jacobian_core(x_tree, q, contactpoint, idcontact, flag_contact,
        parent, jtype, jaxis, nb, nc, nf)
    jcdot_qdot = calc_contact_jdot_qdot_core(x_tree, q, qdot, contactpoint, idcontact, flag_contact,
        parent, jtype, jaxis, nb, nc, nf)
    tau = jnp.reshape(tau, (-1, 1))
    c = jnp.reshape(c, (-1, 1))
    m = jnp.matmul(jc, jnp.linalg.solve(h, jnp.transpose(jc)))
    d0 = jnp.matmul(jc, jnp.linalg.solve(h, tau - c))
    # M = jnp.matmul(jnp.matmul(Jc, Hinv), jnp.transpose(Jc))
    # d0 = jnp.matmul(np.matmul(Jc, Hinv), tau - C)
    d = jnp.add(d0, jcdot_qdot)
    fqp = -jnp.linalg.solve(m, d)
    flcp = jnp.matmul(jnp.transpose(jc), fqp)
    flcp = jnp.reshape(flcp, (-1,))
    return flcp, fqp


def calc_contact_force_direct(model: dict, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray, flag_contact: np.ndarray):
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
    hh = model["hh"]
    cc = model["cc"]
    flag_recalc = 1
    fqp = np.empty((0, 1))
    flcp = np.empty((0, 1))
    while flag_recalc:
        if np.sum(flag_contact) == 0:
            fqp = np.zeros((nc*nf, 1))
            flcp = np.zeros((nb, 1))
            fc = np.zeros((3*nc,))
            fcqp = np.zeros((3*nc,))
            fcpd = np.zeros((3*nc,))
            break
        flcp, fqp = calc_contact_force_direct_core(
            x_tree, q, qdot, contactpoint, hh, tau, cc, idcontact, flag_contact,
            parent, jtype, jaxis, nb, nc, nf)
        # Check whether the Fz is positive
        flag_contact, flag_recalc = check_contact_force(model, flag_contact, fqp)
    # Calculate contact force from PD controller
    # fpd = calc_contact_force_pd(model, q, qdot, flag_contact)
    # fc, fcqp, fcpd = get_contact_force(model, fqp, fpd, flag_contact)
    fpd = np.zeros((3*nc, 1))
    fc, fcqp, fcpd = get_contact_force(model, fqp, fpd, flag_contact)
    return flcp, fqp, fc, fcqp, fcpd
