import numpy as np
from jaxRBDL.Model.JointModel import JointModel
from jaxRBDL.Math.Xtrans import Xtrans
import jax.numpy as jnp
from jax.api import jit
from functools import partial

@partial(jit, static_argnums=(1, 2, 3, 4))
def CalcPointVelocityCore(Xtree, parent, jtype, jaxis, body_id, q, qdot, point_pos):
    X0 = []
    Xup = []
    S = []
    v = []

    for i in range(body_id):
        XJ, Si = JointModel(jtype[i], jaxis[i], q[i])
        S.append(Si)
        vJ = jnp.multiply(S[i], qdot[i])
        Xup = jnp.matmul(XJ, Xtree[i])
        if parent[i] == 0:
            v.append(vJ)
            X0.append(Xup)
        else:
            v.append(jnp.add(jnp.matmul(Xup, v[parent[i] - 1]), vJ))
            X0.append(jnp.matmul(Xup, X0[parent[i] - 1]))
    
    XT_point = Xtrans(point_pos)
    X0_point = jnp.matmul(XT_point,  X0[body_id-1])
    vel_spatial = jnp.matmul(XT_point, v[body_id-1])
    vel = jnp.matmul(jnp.transpose(X0_point[0:3,0:3]), vel_spatial[3:6])

    return vel


def CalcPointVelocity(model: dict, q: np.ndarray, qdot: np.ndarray, body_id: int, point_pos: np.ndarray)->np.ndarray:
    q = q.flatten()
    qdot = qdot.flatten()
    point_pos = point_pos.flatten()
    jtype = model['jtype']
    jaxis = model['jaxis']
    parent = model['parent']
    Xtree = model['Xtree']
    vel = CalcPointVelocityCore(Xtree, tuple(parent), tuple(jtype), jaxis, body_id, q, qdot, point_pos)
    return vel