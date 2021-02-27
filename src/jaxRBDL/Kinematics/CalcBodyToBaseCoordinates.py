from typing import Pattern
from jax.api import jit
import numpy as np
import jax.numpy as jnp
from jaxRBDL.Model.JointModel import JointModel
from jaxRBDL.Math.Xtrans import Xtrans
from jaxRBDL.Kinematics.TransformToPosition import TransformToPosition
from functools import partial

@partial(jit, static_argnums=(1, 2, 3, 4))
def CalcBodyToBaseCoordinatesCore(Xtree, parent, jtype, jaxis, body_id, q, point_pos):
    X0 = []
    for i in range(body_id):
        XJ, _ = JointModel(jtype[i], jaxis[i], q[i])
        Xup = jnp.matmul(XJ, Xtree[i])
        if parent[i] == 0:
            X0.append(Xup)
        else:
            X0.append(jnp.matmul(Xup, X0[parent[i] - 1]))
    
    XT_point = Xtrans(point_pos)
    X0_point =  jnp.matmul(XT_point, X0[body_id - 1])
    pos = TransformToPosition(X0_point)
    return pos


def CalcBodyToBaseCoordinates(model: dict, q: np.ndarray, body_id: int, point_pos: np.ndarray)->np.ndarray:
    q = q.flatten()
    point_pos = point_pos.flatten()
    jtype = model['jtype']
    jaxis = model['jaxis']
    parent = model['parent']
    Xtree = model['Xtree']

    pos = CalcBodyToBaseCoordinatesCore(Xtree, tuple(parent), tuple(jtype), tuple(jaxis), body_id, q, point_pos)
    return pos
