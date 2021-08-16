from typing import Pattern
from jax.api import jit
import numpy as np
import jax.numpy as jnp
from jbdl.rbdl.model import joint_model
from jbdl.rbdl.math import x_trans
from jbdl.rbdl.kinematics import transform_to_position
from functools import partial
from jbdl.rbdl.utils import xyz2int

@partial(jit, static_argnums=(1, 2, 3, 4))
def calc_body_to_base_coordinates_core(x_tree, parent, jtype, jaxis, body_id, q, point_pos):
    X0 = []
    for i in range(body_id):
        XJ, _ = joint_model(jtype[i], jaxis[i], q[i])
        Xup = jnp.matmul(XJ, x_tree[i])
        if parent[i] == 0:
            X0.append(Xup)
        else:
            X0.append(jnp.matmul(Xup, X0[parent[i] - 1]))
    
    XT_point = x_trans(point_pos)
    X0_point =  jnp.matmul(XT_point, X0[body_id - 1])
    pos = transform_to_position(X0_point)
    return pos


def calc_body_to_base_coordinates(model: dict, q: np.ndarray, body_id: int, point_pos: np.ndarray)->np.ndarray:
    q = q.flatten()
    point_pos = point_pos.flatten()
    jtype = model['jtype']
    jaxis = xyz2int(model['jaxis'])
    parent = model['parent']
    x_tree = model['x_tree']
    pos = calc_body_to_base_coordinates_core(x_tree, tuple(parent), tuple(jtype), tuple(jaxis), body_id, q, point_pos)
    return pos
