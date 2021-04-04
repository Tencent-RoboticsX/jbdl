from typing import Pattern
from jax.api import jit
import numpy as np
import jax.numpy as jnp
from jaxRBDL.Model.JointModel import JointModel
from jaxRBDL.Math.Xtrans import Xtrans
from jaxRBDL.Kinematics import transform_to_position
from functools import partial

@partial(jit, static_argnums=(1, 2, 3, 4))
def calc_body_to_base_coordinates_core(x_tree, parent, jtype, jaxis, body_id, q, point_pos):
    x0 = []
    for i in range(body_id):
        x_joint, _ = JointModel(jtype[i], jaxis[i], q[i])
        x_up = jnp.matmul(x_joint, x_tree[i])
        if parent[i] == 0:
            x0.append(x_up)
        else:
            x0.append(jnp.matmul(x_up, x0[parent[i] - 1]))
    
    point_trans = Xtrans(point_pos)
    x_end_point =  jnp.matmul(point_trans, x0[body_id - 1])
    pos = transform_to_position(x_end_point)
    return pos


def calc_body_to_base_coordinates(model: dict, q: np.ndarray, body_id: int, point_pos: np.ndarray)->np.ndarray:
    q = q.flatten()
    point_pos = point_pos.flatten()
    jtype = model['jtype']
    jaxis = model['jaxis']
    parent = model['parent']
    Xtree = model['Xtree']
    pos = calc_body_to_base_coordinates_core(Xtree, tuple(parent), tuple(jtype), tuple(jaxis), body_id, q, point_pos)
    return pos
