import numpy as np
from numpy.core.shape_base import hstack
from jbdl.rbdl.contact import calc_contact_jacobian, calc_contact_jacobian_core
from jbdl.rbdl.contact.calc_contact_jacobian import calc_contact_jacobian_extend_core
from numpy.linalg import matrix_rank
import jax.numpy as jnp
from jax.api import jit
from functools import partial
from jbdl.rbdl.utils import xyz2int

# @partial(jit, static_argnums=(5, 6, 7, 8, 9, 10, 11, 12, 13))
def impulsive_dynamics_core(x_tree, q, qdot, contactpoint, H, idcontact, flag_contact, parent, jtype, jaxis, NB, nc, nf, rankJc):
    Jc = calc_contact_jacobian_core(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, nc, nf)

    # Calcualet implusive dynamics for qdot after impulsive
    A0 = jnp.hstack([H, -jnp.transpose(Jc)])
    A1 = jnp.hstack([Jc, jnp.zeros((rankJc, rankJc))])
    A = jnp.vstack([A0, A1])

    b0 = jnp.matmul(H, qdot)
    b1 = jnp.zeros((rankJc, ))
    b = jnp.hstack([b0, b1])

    QdotI = jnp.linalg.solve(A, b)
    qdot_impulse = jnp.reshape(QdotI[0:NB], (-1, 1))
    return qdot_impulse

def impulsive_dynamics_extend_core(x_tree, q, qdot, contactpoint, H, idcontact, flag_contact, parent, jtype, jaxis, NB, nc, nf):
    Jc = calc_contact_jacobian_extend_core(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, nc, nf)
    rankJc = nf * nc
    # Calcualet implusive dynamics for qdot after impulsive
    A0 = jnp.hstack([H, -jnp.transpose(Jc)])
    A1 = jnp.hstack([Jc, jnp.zeros((rankJc, rankJc))])
    A = jnp.vstack([A0, A1])

    b0 = jnp.matmul(H, qdot)
    b1 = jnp.zeros((rankJc, ))
    b = jnp.hstack([b0, b1])

    QdotI, residuals, rank, s  = jnp.linalg.lstsq(A, b)
    qdot_impulse = jnp.reshape(QdotI[0:NB], (-1, 1))
    return qdot_impulse





def impulsive_dynamics(model: dict, q: np.ndarray, qdot: np.ndarray, flag_contact:np.ndarray)->np.ndarray:
    q = q.flatten()
    qdot = qdot.flatten()
    nc = int(model["nc"])
    NB = int(model["NB"])
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
    rankJc = int(np.sum( [1 for item in flag_contact if item != 0]) * model["nf"])

    qdot_impulse = impulsive_dynamics_core(x_tree, q, qdot, contactpoint, H, idcontact, flag_contact, parent, jtype, jaxis, NB, nc, nf, rankJc)
    return qdot_impulse


# def impulsive_dynamics(model: dict, q: np.ndarray, qdot: np.ndarray, flag_contact:np.ndarray)->np.ndarray:
#     q = q.flatten()
#     qdot = qdot.flatten()
#     NB = int(model["NB"])
#     H = model["H"]


#     Jc = calc_contact_jacobian(model, q, flag_contact)
#     rankJc = matrix_rank(Jc)

#     # Calcualet implusive dynamics for qdot after impulsive
#     A0 = np.hstack([H, -np.transpose(Jc)])
#     A1 = np.hstack([Jc, np.zeros((rankJc, rankJc))])
#     A = np.vstack([A0, A1])
#     b0 = np.matmul(H, qdot)
#     b1 = np.zeros((rankJc, ))
#     b = np.hstack([b0, b1])
#     QdotI = np.linalg.solve(A, b)
#     qdot_impulse = QdotI[0:NB]
#     qdot_impulse = qdot_impulse.reshape(-1, 1)
#     return qdot_impulse
 
 
