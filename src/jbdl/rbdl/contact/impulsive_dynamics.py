import numpy as np
from jbdl.rbdl.contact import calc_contact_jacobian_core
from jbdl.rbdl.contact.calc_contact_jacobian import calc_contact_jacobian_extend_core
import jax.numpy as jnp
from jbdl.rbdl.utils import xyz2int


# @partial(jit, static_argnums=(5, 6, 7, 8, 9, 10, 11, 12, 13))
def impulsive_dynamics_core(
    x_tree, q, qdot, contactpoint, h, idcontact, flag_contact,
    parent, jtype, jaxis, nb, nc, nf, rank_jc):

    jc = calc_contact_jacobian_core(
        x_tree, q, contactpoint, idcontact, flag_contact,
        parent, jtype, jaxis, nb, nc, nf)

    # Calcualet implusive dynamics for qdot after impulsive
    a0 = jnp.hstack([h, -jnp.transpose(jc)])
    a1 = jnp.hstack([jc, jnp.zeros((rank_jc, rank_jc))])
    a = jnp.vstack([a0, a1])

    b0 = jnp.matmul(h, qdot)
    b1 = jnp.zeros((rank_jc, ))
    b = jnp.hstack([b0, b1])

    qdot_inertia = jnp.linalg.solve(a, b)
    qdot_impulse = jnp.reshape(qdot_inertia[0:nb], (-1, 1))
    return qdot_impulse

def impulsive_dynamics_extend_core(
    x_tree, q, qdot, contactpoint, h, idcontact, flag_contact,
    parent, jtype, jaxis, nb, nc, nf):

    jc = calc_contact_jacobian_extend_core(
        x_tree, q, contactpoint, idcontact, flag_contact,
        parent, jtype, jaxis, nb, nc, nf)
    rank_jc = nf * nc

    # Calcualet implusive dynamics for qdot after impulsive
    a0 = jnp.hstack([h, -jnp.transpose(jc)])
    a1 = jnp.hstack([jc, jnp.zeros((rank_jc, rank_jc))])
    a = jnp.vstack([a0, a1])

    b0 = jnp.matmul(h, qdot)
    b1 = jnp.zeros((rank_jc, ))
    b = jnp.hstack([b0, b1])

    qdot_inertia, _, _, _  = jnp.linalg.lstsq(a, b)
    qdot_impulse = jnp.reshape(qdot_inertia[0:nb], (-1, 1))
    return qdot_impulse





def impulsive_dynamics(
    model: dict, q: np.ndarray, qdot: np.ndarray, flag_contact:np.ndarray) -> np.ndarray:

    q = q.flatten()
    qdot = qdot.flatten()
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
    h = model["h"]
    rank_jc = int(np.sum( [1 for item in flag_contact if item != 0]) * model["nf"])

    qdot_impulse = impulsive_dynamics_core(x_tree, q, qdot, contactpoint, h, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, rank_jc)
    return qdot_impulse
