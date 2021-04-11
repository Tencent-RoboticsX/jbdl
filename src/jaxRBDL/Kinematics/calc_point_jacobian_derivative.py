import numpy as np
import jax.numpy as jnp
from jax.api import jit
from jaxRBDL.Model import joint_model
from jaxRBDL.Math.Xtrans import Xtrans
from jaxRBDL.Math.CrossMotionSpace import CrossMotionSpace
from jaxRBDL.Math.InverseMotionSpace import InverseMotionSpace
from functools import partial


@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def calc_point_jacobian_derivative_core(Xtree, parent, jtype, jaxis, body_id, NB, q, qdot, point_pos):

    S = []
    Xup = []
    X0 = []
    v = []

    for i in range(body_id):
        XJ, Si = joint_model(jtype[i], jaxis[i], q[i])
        S.append(Si)
        vJ = jnp.multiply(S[i], qdot[i])
        Xup.append(jnp.matmul(XJ, Xtree[i]))
        if parent[i] == 0:
            v.append(vJ)
            X0.append(Xup[i])
        else:
            v.append(jnp.add(jnp.matmul(Xup[i], v[parent[i]-1]), vJ))
            X0.append(jnp.matmul(Xup[i], X0[parent[i]-1]))

    XT_point = Xtrans(point_pos)
    X0_point = jnp.matmul(XT_point, X0[body_id-1])
    v_point = jnp.matmul(XT_point, v[body_id-1])

    BJ = jnp.zeros((6, NB))
    dBJ = jnp.zeros((6, NB))
    id_p = id =  body_id - 1
    Xe = jnp.zeros((NB, 6, 6))

    while id_p != -1:
        if id_p == body_id - 1:
            Xe = Xe.at[id_p,...].set(jnp.matmul(XT_point, Xup[id_p]))
            BJ = BJ.at[:,[id_p,]].set(jnp.matmul(XT_point, S[id_p]))
            dBJ = dBJ.at[:,[id_p,]].set(jnp.matmul(jnp.matmul(CrossMotionSpace(jnp.matmul(XT_point, v[id_p]) - v_point), XT_point), S[id_p]))
        else:
            Xe = Xe.at[id_p,...].set(jnp.matmul(Xe[id, ...], Xup[id_p]))
            BJ = BJ.at[:,[id_p,]].set(jnp.matmul(Xe[id, ...], S[id_p]))
            dBJ = dBJ.at[:,[id_p,]].set(jnp.matmul(jnp.matmul(CrossMotionSpace(jnp.matmul(Xe[id, ...], v[id_p]) - v_point), Xe[id,...]), S[id_p]) )       
        id = id_p
        id_p = parent[id] - 1
    X0 = InverseMotionSpace(X0_point)
    E0 = jnp.vstack([jnp.hstack([X0[0:3,0:3], jnp.zeros((3, 3))]), jnp.hstack([jnp.zeros((3, 3)), X0[0:3, 0:3]])])
    dE0 = jnp.matmul(CrossMotionSpace(jnp.matmul(X0,v_point)), E0)
    E0 = E0[0:3, 0:3]
    dE0 = dE0[0:3,0:3]
    JDot = jnp.matmul(jnp.matmul(dE0, jnp.hstack([jnp.zeros((3,3)), jnp.eye(3)])), BJ) \
        + jnp.matmul(jnp.matmul(E0, jnp.hstack([jnp.zeros((3,3)), jnp.eye(3)])), dBJ)

    return JDot



def calc_point_jacobian_derivative(model: dict, q: np.ndarray, qdot: np.ndarray, body_id: int, point_pos: np.ndarray)->np.ndarray:
    q = q.flatten()
    qdot = qdot.flatten()
    point_pos = point_pos.flatten()
    jtype = model['jtype']
    jaxis = model['jaxis']
    parent = model['parent']
    NB = model["NB"]
    Xtree = model['Xtree']

    JDot = calc_point_jacobian_derivative_core(Xtree, tuple(parent), tuple(jtype), jaxis, body_id, NB, q, qdot, point_pos)
    return JDot
