import numpy as np
import jax.numpy as jnp
from jaxRBDL.Math.CrossMotionSpace import CrossMotionSpace
from jaxRBDL.Math.Xtrans import Xtrans
from jaxRBDL.Model.JointModel import JointModel
from jax.api import jit
from functools import partial

@partial(jit, static_argnums=(1, 2, 3, 4))
def calc_point_acceleration_core(Xtree, parent, jtype, jaxis, body_id, q, qdot, qddot, point_pos):
    Xup = []
    v = []
    avp = []
    X0 = []

    for i in range(body_id):
        XJ, Si = JointModel(jtype[i], jaxis[i], q[i])
        vJ = jnp.multiply(Si,  qdot[i])
        Xup.append(jnp.matmul(XJ, Xtree[i]))
        if parent[i] == 0:
            v.append(vJ)
            avp.append(jnp.matmul(Xup[i], jnp.zeros((6, 1))))
            X0.append(Xup[i])
        else:
            v.append(jnp.matmul(Xup[i], v[parent[i]-1]) + vJ)
            avp.append(jnp.matmul(Xup[i], avp[parent[i] - 1]) + jnp.multiply(Si, qddot[i]) + jnp.matmul(CrossMotionSpace(v[i]), vJ))
            X0.append(jnp.matmul(Xup[i], X0[parent[i]-1]))

    E_point = X0[body_id-1][0:3,0:3]
    XT_point = Xtrans(point_pos)
    vel_p = jnp.matmul(XT_point, v[body_id-1])
    avp_p = jnp.matmul(XT_point, avp[body_id-1])
    
    acc = jnp.matmul(jnp.transpose(E_point), avp_p[3:6]) + \
        jnp.reshape(jnp.cross(jnp.squeeze(jnp.matmul(jnp.transpose(E_point), vel_p[0:3])), jnp.squeeze(jnp.matmul(jnp.transpose(E_point), vel_p[3:6]))), (3, 1))

    return acc




def calc_point_acceleration(model: dict, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray, body_id: int, point_pos: np.ndarray)->np.ndarray:
    q = q.flatten()
    qdot = qdot.flatten()
    qddot = qddot.flatten()
    point_pos = point_pos.flatten()
    jtype = model['jtype']
    jaxis = model['jaxis']
    parent = model['parent']
    Xtree = model['Xtree']

    acc = calc_point_acceleration_core(Xtree, tuple(parent), tuple(jtype), jaxis, body_id, q, qdot, qddot, point_pos)
    return acc