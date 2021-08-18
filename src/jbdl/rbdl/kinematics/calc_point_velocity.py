from functools import partial
import numpy as np
from jbdl.rbdl.model import joint_model
from jbdl.rbdl.math import x_trans
import jax.numpy as jnp
from jax.api import jit
from jbdl.rbdl.utils import xyz2int


@partial(jit, static_argnums=(1, 2, 3, 4))
def calc_point_velocity_core(x_tree, parent, jtype, jaxis, body_id, q, qdot, point_pos):
    x0 = []
    x_up = []
    s = []
    v = []

    for i in range(body_id):
        xj, si = joint_model(jtype[i], jaxis[i], q[i])
        s.append(si)
        vj = jnp.multiply(s[i], qdot[i])
        x_up = jnp.matmul(xj, x_tree[i])
        if parent[i] == 0:
            v.append(vj)
            x0.append(x_up)
        else:
            v.append(jnp.add(jnp.matmul(x_up, v[parent[i] - 1]), vj))
            x0.append(jnp.matmul(x_up, x0[parent[i] - 1]))

    x_final_point = x_trans(point_pos)
    x0_point = jnp.matmul(x_final_point,  x0[body_id-1])
    vel_spatial = jnp.matmul(x_final_point, v[body_id-1])
    vel = jnp.matmul(jnp.transpose(x0_point[0:3, 0:3]), vel_spatial[3:6])
    return vel


def calc_point_velocity(
    model: dict, q: np.ndarray, qdot: np.ndarray,
    body_id: int, point_pos: np.ndarray) -> np.ndarray:

    q = q.flatten()
    qdot = qdot.flatten()
    point_pos = point_pos.flatten()
    jtype = tuple(model['jtype'])
    jaxis = xyz2int(model['jaxis'])
    parent = tuple(model['parent'])
    x_tree = model['x_tree']
    vel = calc_point_velocity_core(x_tree, tuple(parent), tuple(jtype), jaxis, body_id, q, qdot, point_pos)
    return vel
