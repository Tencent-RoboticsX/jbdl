import numpy as np
import jax.numpy as jnp
from jax.api import jit
from jaxRBDL.model import joint_model
from jaxRBDL.math import cross_motion_space, cross_force_space
from functools import partial

@partial(jit, static_argnums=(2, 3, 4, 5))
def inverse_dynamics_core(Xtree, I, parent, jtype, jaxis, NB, q, qdot, qddot, a_grav):
    S = []
    Xup = []
    v = []
    avp = []
    fvp = []

    for i in range(NB):
        XJ, Si = joint_model(jtype[i], jaxis[i], q[i])
        S.append(Si)
        vJ = jnp.multiply(S[i], qdot[i])
        Xup.append(jnp.matmul(XJ, Xtree[i]))
        if parent[i] == 0:
            v.append(vJ)
            avp.append(jnp.matmul(Xup[i], -a_grav) + S[i] * qddot[i])
        else:
            v.append(jnp.matmul(Xup[i], v[parent[i] - 1])+ vJ)
            avp.append(jnp.matmul(Xup[i], avp[parent[i] - 1]) + jnp.multiply(S[i], qddot[i]) + jnp.matmul(cross_motion_space(v[i]), vJ))
        fvp.append(jnp.matmul(I[i], avp[i]) + jnp.matmul(jnp.matmul(cross_force_space(v[i]), I[i]), v[i]))

    tau = [0.0] * NB

    for i in range(NB-1, -1, -1):
        tau[i] = jnp.squeeze(jnp.matmul(jnp.transpose(S[i]), fvp[i]))
        if parent[i] != 0:
            fvp[parent[i] - 1] = fvp[parent[i] - 1] + jnp.matmul(jnp.transpose(Xup[i]), fvp[i])
    tau = jnp.reshape(jnp.array(tau), (NB, 1))

    return tau 


def inverse_dynamics(model, q, qdot, qddot):
    
    a_grav = model["a_grav"]
    qdot = qdot.flatten()
    qddot = qddot.flatten()
    NB = model["NB"] 

    jtype = model["jtype"]
    jaxis = model["jaxis"]
    parent = model["parent"]
    Xtree = model["Xtree"]
    I = model["I"]

    tau = inverse_dynamics_core(Xtree, I, tuple(parent), tuple(jtype), jaxis, NB, q, qdot, qddot, a_grav)
    return tau 




