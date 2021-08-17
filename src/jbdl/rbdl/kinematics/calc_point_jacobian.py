from functools import partial
import numpy as np
import jax.numpy as jnp
from jbdl.rbdl.model import joint_model
from jbdl.rbdl.math import x_trans, inverse_motion_space
from jax.api import jit
from jbdl.rbdl.utils import xyz2int


@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def calc_point_jacobian_core(x_tree, parent, jtype, jaxis, nb, body_id, q, point_pos):
    s = []
    x_up = []
    x0 = []

    for i in range(body_id):
        xj, si = joint_model(jtype[i], jaxis[i], q[i])
        s.append(si)
        x_up.append(jnp.matmul(xj, x_tree[i]))
        if parent[i] == 0:
            x0.append(x_up[i])
        else:
            x0.append(jnp.matmul(x_up[i], x0[parent[i]-1]))

    x_final_point = x_trans(point_pos)
    x0_point = jnp.matmul(x_final_point, x0[body_id-1])

    jp = body_id - 1
    bj = jnp.zeros((6, nb))
    while jp != -1:
        xe = jnp.matmul(x0_point, inverse_motion_space(x0[jp]))
        bj = bj.at[:, [jp, ]].set(jnp.matmul(xe, s[jp]))
        jp = parent[jp] - 1

    e0 = jnp.transpose(x0_point[0:3, 0:3])
    jac = jnp.matmul(jnp.matmul(e0, jnp.hstack([ jnp.zeros((3, 3)), jnp.eye(3)])), bj)

    return jac


def calc_point_jacobian(model: dict, q: np.ndarray, body_id: int, point_pos: np.ndarray) -> np.ndarray:
    q = q.flatten()
    point_pos = point_pos.flatten()
    jtype = tuple(model['jtype'])
    jaxis = xyz2int(model['jaxis'])
    parent = tuple(model['parent'])
    nb = model["nb"]
    x_tree = model['x_tree']

    jac = calc_point_jacobian_core(x_tree, tuple(parent), tuple(jtype), jaxis, nb, body_id, q, point_pos)
    return jac
