from functools import partial
import numpy as np
import jax.numpy as jnp
from jbdl.rbdl.math import cross_motion_space, x_trans
from jbdl.rbdl.model import joint_model
from jax.api import jit
from jbdl.rbdl.utils import xyz2int


@partial(jit, static_argnums=(1, 2, 3, 4))
def calc_point_acceleration_core(x_tree, parent, jtype, jaxis, body_id, q, qdot, qddot, point_pos):
    x_up = []
    v = []
    avp = []
    x0 = []

    for i in range(body_id):
        xj, si = joint_model(jtype[i], jaxis[i], q[i])
        vj = jnp.multiply(si,  qdot[i])
        x_up.append(jnp.matmul(xj, x_tree[i]))
        if parent[i] == 0:
            v.append(vj)
            avp.append(jnp.matmul(x_up[i], jnp.zeros((6, 1))))
            x0.append(x_up[i])
        else:
            v.append(jnp.matmul(x_up[i], v[parent[i]-1]) + vj)
            avp.append(jnp.matmul(x_up[i], avp[parent[i] - 1]) + jnp.multiply(si, qddot[i]) + jnp.matmul(cross_motion_space(v[i]), vj))
            x0.append(jnp.matmul(x_up[i], x0[parent[i]-1]))

    e_point = x0[body_id-1][0:3,0:3]
    x_final_point = x_trans(point_pos)
    vel_p = jnp.matmul(x_final_point, v[body_id-1])
    avp_p = jnp.matmul(x_final_point, avp[body_id-1])
    
    acc = jnp.matmul(jnp.transpose(e_point), avp_p[3:6]) + \
        jnp.reshape(jnp.cross(jnp.squeeze(jnp.matmul(jnp.transpose(e_point), vel_p[0:3])),
        jnp.squeeze(jnp.matmul(jnp.transpose(e_point), vel_p[3:6]))), (3, 1))

    return acc


def calc_point_acceleration(
    model: dict, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray,
    body_id: int, point_pos: np.ndarray) -> np.ndarray:
    q = q.flatten()
    qdot = qdot.flatten()
    qddot = qddot.flatten()
    point_pos = point_pos.flatten()
    jtype = tuple(model['jtype'])
    jaxis = xyz2int(model['jaxis'])
    parent = tuple(model['parent'])
    x_tree = model['x_tree']

    acc = calc_point_acceleration_core(
        x_tree, tuple(parent), tuple(jtype), jaxis, body_id, q, qdot, qddot, point_pos)
    return acc
