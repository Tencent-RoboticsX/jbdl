import numpy as np
from numpy.core.shape_base import hstack
from jaxRBDL.Contact.CalcContactJacobian import CalcContactJacobian, CalcContactJacobianCore
from numpy.linalg import matrix_rank
import jax.numpy as jnp
from jax.api import jit
from functools import partial

@partial(jit, static_argnums=(5, 6, 7, 8, 9, 10, 11, 12, 13))
def ImpulsiveDynamicsCore(Xtree, q, qdot, contactpoint, H, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, rankJc):
    Jc = CalcContactJacobianCore(Xtree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)

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





def ImpulsiveDynamics(model: dict, q: np.ndarray, qdot: np.ndarray, flag_contact:np.ndarray)->np.ndarray:
    q = q.flatten()
    qdot = qdot.flatten()
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
    rankJc = np.sum( [1 for item in flag_contact if item != 0]) * model["nf"]

    qdot_impulse = ImpulsiveDynamicsCore(Xtree, q, qdot, contactpoint, H, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, rankJc)
    return qdot_impulse


# def ImpulsiveDynamics(model: dict, q: np.ndarray, qdot: np.ndarray, flag_contact:np.ndarray)->np.ndarray:
#     q = q.flatten()
#     qdot = qdot.flatten()
#     NB = int(model["NB"])
#     H = model["H"]


#     Jc = CalcContactJacobian(model, q, flag_contact)
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
 
 
