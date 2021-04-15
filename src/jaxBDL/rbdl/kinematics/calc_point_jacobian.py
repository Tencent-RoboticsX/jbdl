import numpy as np
import jax.numpy as jnp
from jaxBDL.rbdl.model import joint_model
from jaxBDL.rbdl.math import Xtrans, inverse_motion_space
from jax.api import jit
from functools import partial

@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def calc_point_jacobian_core(Xtree, parent, jtype, jaxis, NB, body_id, q, point_pos):
    S = []
    Xup = []
    X0 = []

    for i in range(body_id):
        XJ, Si = joint_model(jtype[i], jaxis[i], q[i])
        S.append(Si)
        Xup.append(jnp.matmul(XJ, Xtree[i]))
        if parent[i] == 0:
            X0.append(Xup[i])
        else:
            X0.append(jnp.matmul(Xup[i], X0[parent[i]-1]))

    XT_point = Xtrans(point_pos)
    X0_point = jnp.matmul(XT_point, X0[body_id-1])

    j_p = body_id - 1
    BJ = jnp.zeros((6, NB))
    while j_p != -1:
        Xe = jnp.matmul(X0_point, inverse_motion_space(X0[j_p]))
        BJ = BJ.at[:, [j_p, ]].set(jnp.matmul(Xe, S[j_p]))
        j_p = parent[j_p] - 1

    E0 = jnp.transpose(X0_point[0:3, 0:3])
    J = jnp.matmul(jnp.matmul(E0, jnp.hstack([ jnp.zeros((3, 3)), jnp.eye(3)])), BJ)

    return J


def calc_point_jacobian(model: dict, q: np.ndarray, body_id: int, point_pos: np.ndarray)->np.ndarray:
    q = q.flatten()
    point_pos = point_pos.flatten()
    jtype = model['jtype']
    jaxis = model['jaxis']
    parent = model['parent']
    NB = model["NB"]
    Xtree = model['Xtree']

    J = calc_point_jacobian_core(Xtree, tuple(parent), tuple(jtype), jaxis, NB, body_id, q, point_pos)
    return J
