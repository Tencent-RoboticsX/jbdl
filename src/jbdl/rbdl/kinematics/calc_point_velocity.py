import numpy as np
from jbdl.rbdl.model import joint_model
from jbdl.rbdl.math import x_trans
import jax.numpy as jnp
from jax.api import jit
from functools import partial
from jbdl.rbdl.utils import xyz2int

@partial(jit, static_argnums=(1, 2, 3, 4))
def calc_point_velocity_core(x_tree, parent, jtype, jaxis, body_id, q, qdot, point_pos):
    X0 = []
    Xup = []
    S = []
    v = []

    for i in range(body_id):
        XJ, Si = joint_model(jtype[i], jaxis[i], q[i])
        S.append(Si)
        vJ = jnp.multiply(S[i], qdot[i])
        Xup = jnp.matmul(XJ, x_tree[i])
        if parent[i] == 0:
            v.append(vJ)
            X0.append(Xup)
        else:
            v.append(jnp.add(jnp.matmul(Xup, v[parent[i] - 1]), vJ))
            X0.append(jnp.matmul(Xup, X0[parent[i] - 1]))
    
    XT_point = x_trans(point_pos)
    X0_point = jnp.matmul(XT_point,  X0[body_id-1])
    vel_spatial = jnp.matmul(XT_point, v[body_id-1])
    vel = jnp.matmul(jnp.transpose(X0_point[0:3,0:3]), vel_spatial[3:6])

    return vel


def calc_point_velocity(model: dict, q: np.ndarray, qdot: np.ndarray, body_id: int, point_pos: np.ndarray)->np.ndarray:
    q = q.flatten()
    qdot = qdot.flatten()
    point_pos = point_pos.flatten()
    jtype = tuple(model['jtype'])
    jaxis = xyz2int(model['jaxis'])
    parent = tuple(model['parent'])
    x_tree = model['x_tree']
    vel = calc_point_velocity_core(x_tree, tuple(parent), tuple(jtype), jaxis, body_id, q, qdot, point_pos)
    return vel