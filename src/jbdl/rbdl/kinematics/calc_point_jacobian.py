import numpy as np
import jax.numpy as jnp
from jbdl.rbdl.model import joint_model
from jbdl.rbdl.math import x_trans, inverse_motion_space
from jax.api import jit
from functools import partial
from jbdl.rbdl.utils import xyz2int

@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def calc_point_jacobian_core(x_tree, parent, jtype, jaxis, nb, body_id, q, point_pos):
    S = []
    Xup = []
    X0 = []

    for i in range(body_id):
        XJ, Si = joint_model(jtype[i], jaxis[i], q[i])
        S.append(Si)
        Xup.append(jnp.matmul(XJ, x_tree[i]))
        if parent[i] == 0:
            X0.append(Xup[i])
        else:
            X0.append(jnp.matmul(Xup[i], X0[parent[i]-1]))

    XT_point = x_trans(point_pos)
    X0_point = jnp.matmul(XT_point, X0[body_id-1])

    j_p = body_id - 1
    BJ = jnp.zeros((6, nb))
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
    jtype = tuple(model['jtype'])
    jaxis = xyz2int(model['jaxis'])
    parent = tuple(model['parent'])
    nb = model["nb"]
    x_tree = model['x_tree']

    J = calc_point_jacobian_core(x_tree, tuple(parent), tuple(jtype), jaxis, nb, body_id, q, point_pos)
    return J
