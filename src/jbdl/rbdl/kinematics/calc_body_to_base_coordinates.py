from functools import partial
from jax.api import jit
import numpy as np
import jax.numpy as jnp
from jbdl.rbdl.model import joint_model
from jbdl.rbdl.math import x_trans
from jbdl.rbdl.kinematics import transform_to_position
from jbdl.rbdl.utils import xyz2int


@partial(jit, static_argnums=(1, 2, 3, 4))
def calc_body_to_base_coordinates_core(x_tree, parent, jtype, jaxis, body_id, q, point_pos):

    x0 = []
    for i in range(body_id):
        xj, _ = joint_model(jtype[i], jaxis[i], q[i])
        x_up = jnp.matmul(xj, x_tree[i])
        if parent[i] == 0:
            x0.append(x_up)
        else:
            x0.append(jnp.matmul(x_up, x0[parent[i] - 1]))

    x_final_point = x_trans(point_pos)
    x0_point =  jnp.matmul(x_final_point, x0[body_id - 1])
    pos = transform_to_position(x0_point)
    return pos


def calc_body_to_base_coordinates(
    model: dict, q: np.ndarray, body_id: int, point_pos: np.ndarray) -> np.ndarray:

    q = q.flatten()
    point_pos = point_pos.flatten()
    jtype = model['jtype']
    jaxis = xyz2int(model['jaxis'])
    parent = model['parent']
    x_tree = model['x_tree']
    pos = calc_body_to_base_coordinates_core(x_tree, tuple(parent), tuple(jtype), tuple(jaxis), body_id, q, point_pos)
    return pos
