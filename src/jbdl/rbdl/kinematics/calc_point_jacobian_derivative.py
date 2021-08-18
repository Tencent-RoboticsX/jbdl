from functools import partial
import numpy as np
import jax.numpy as jnp
from jax.api import jit
from jbdl.rbdl.model import joint_model
from jbdl.rbdl.math import x_trans, cross_motion_space, inverse_motion_space
from jbdl.rbdl.utils import xyz2int


@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def calc_point_jacobian_derivative_core(x_tree, parent, jtype, jaxis, body_id, nb, q, qdot, point_pos):

    s = []
    x_up = []
    x0 = []
    v = []

    for i in range(body_id):
        xj, si = joint_model(jtype[i], jaxis[i], q[i])
        s.append(si)
        vj = jnp.multiply(s[i], qdot[i])
        x_up.append(jnp.matmul(xj, x_tree[i]))
        if parent[i] == 0:
            v.append(vj)
            x0.append(x_up[i])
        else:
            v.append(jnp.add(jnp.matmul(x_up[i], v[parent[i]-1]), vj))
            x0.append(jnp.matmul(x_up[i], x0[parent[i]-1]))

    x_final_point = x_trans(point_pos)
    x0_point = jnp.matmul(x_final_point, x0[body_id-1])
    v_point = jnp.matmul(x_final_point, v[body_id-1])

    bj = jnp.zeros((6, nb))
    dbj = jnp.zeros((6, nb))
    id_p = id =  body_id - 1
    xe = jnp.zeros((nb, 6, 6))

    while id_p != -1:
        if id_p == body_id - 1:
            xe = xe.at[id_p,...].set(
                jnp.matmul(x_final_point, x_up[id_p]))
            bj = bj.at[:,[id_p,]].set(
                jnp.matmul(x_final_point, s[id_p]))
            dbj = dbj.at[:,[id_p,]].set(
                jnp.matmul(jnp.matmul(
                    cross_motion_space(jnp.matmul(x_final_point, v[id_p]) - v_point), x_final_point), s[id_p]))
        else:
            xe = xe.at[id_p,...].set(
                np.matmul(xe[id, ...], x_up[id_p]))
            bj = bj.at[:,[id_p,]].set(
                jnp.matmul(xe[id, ...], s[id_p]))
            dbj = dbj.at[:,[id_p,]].set(
                jnp.matmul(jnp.matmul(
                    cross_motion_space(jnp.matmul(xe[id, ...], v[id_p]) - v_point), xe[id,...]), s[id_p]) )       
        id = id_p
        id_p = parent[id] - 1
    x0 = inverse_motion_space(x0_point)
    e0 = jnp.vstack(
        [jnp.hstack([x0[0:3,0:3],jnp.zeros((3, 3))]),
        jnp.hstack([jnp.zeros((3, 3)), x0[0:3, 0:3]])])
    de0 = jnp.matmul(cross_motion_space(jnp.matmul(x0,v_point)), e0)
    e0 = e0[0:3, 0:3]
    de0 = de0[0:3,0:3]
    jdot = jnp.matmul(jnp.matmul(de0, jnp.hstack([jnp.zeros((3,3)), jnp.eye(3)])), bj) \
        + jnp.matmul(jnp.matmul(e0, jnp.hstack([jnp.zeros((3,3)), jnp.eye(3)])), dbj)

    return jdot


def calc_point_jacobian_derivative(
    model: dict, q: np.ndarray, qdot: np.ndarray,
    body_id: int, point_pos: np.ndarray) -> np.ndarray:
    q = q.flatten()
    qdot = qdot.flatten()
    point_pos = point_pos.flatten()
    jtype = tuple(model['jtype'])
    jaxis = xyz2int(model['jaxis'])
    parent = tuple(model['parent'])
    nb = model["nb"]
    x_tree = model['x_tree']

    jdot = calc_point_jacobian_derivative_core(x_tree, tuple(parent), tuple(jtype), jaxis, body_id, nb, q, qdot, point_pos)
    return jdot

