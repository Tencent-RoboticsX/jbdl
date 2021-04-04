import numpy as np
import jax.numpy as jnp
from jaxRBDL.Model.JointModel import JointModel
from jaxRBDL.Math.Xtrans import Xtrans
from jaxRBDL.Math.InverseMotionSpace import InverseMotionSpace
from jax.api import jit
from functools import partial

@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def calc_point_jacobian_core(x_tree, parent, jtype, jaxis, NB, body_id, q, point_pos):
    S = []
    x_up = []
    x0 = []

    for i in range(body_id):
        x_joint, Si = JointModel(jtype[i], jaxis[i], q[i])
        S.append(Si)
        x_up.append(jnp.matmul(x_joint, x_tree[i]))
        if parent[i] == 0:
            x0.append(x_up[i])
        else:
            x0.append(jnp.matmul(x_up[i], x0[parent[i]-1]))

    point_trans = Xtrans(point_pos)
    x_end_point = jnp.matmul(point_trans, x0[body_id-1])

    j_p = body_id - 1
    BJ = jnp.zeros((6, NB))
    while j_p != -1:
        Xe = jnp.matmul(x_end_point, InverseMotionSpace(x0[j_p]))
        BJ = BJ.at[:, [j_p, ]].set(jnp.matmul(Xe, S[j_p]))
        j_p = parent[j_p] - 1

    E0 = jnp.transpose(x_end_point[0:3, 0:3])
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
