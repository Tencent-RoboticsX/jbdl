import numpy as np
import jax.numpy as jnp
from jaxRBDL.Model.JointModel import JointModel
from jaxRBDL.Math.CrossMotionSpace import CrossMotionSpace
from jaxRBDL.Math.CrossForceSpace import CrossForceSpace

def InverseDynamics(model, q, qdot, qddot):
    
    a_grav = model["a_grav"].reshape(6, 1)
    qdot = qdot.flatten()
    qddot = qddot.flatten()
    NB = int(model["NB"])

    jtype = model["jtype"].flatten()
    jaxis = model['jaxis']
    parent = model['parent'].flatten().astype(int)

    try:
        Xtree = np.squeeze(model['Xtree'], axis=0)
        I =  np.squeeze(model['I'], axis=0)
    except:
        Xtree = model['Xtree']
        I = model['I']

    S = []
    Xup = []
    v = []
    avp = []
    fvp = []

    for i in range(NB):
        XJ, Si = JointModel(jtype[i], jaxis[i], q[i])
        S.append(Si)
        vJ = jnp.multiply(S[i], qdot[i])
        Xup.append(jnp.matmul(XJ, Xtree[i]))
        if parent[i] == 0:
            v.append(vJ)
            avp.append(jnp.matmul(Xup[i], -a_grav))
        else:
            v.append(jnp.matmul(Xup[i], v[parent[i] - 1])+ vJ)
            avp.append(jnp.matmul(Xup[i], avp[parent[i] - 1]) + jnp.multiply(S[i], qddot[i]) + jnp.matmul(CrossMotionSpace(v[i]), vJ))
        fvp.append(jnp.matmul(I[i], avp[i]) + jnp.matmul(jnp.matmul(CrossForceSpace(v[i]), I[i]), v[i]))

    tau = [0.0] * NB

    for i in range(NB-1, -1, -1):
        tau[i] = jnp.squeeze(jnp.matmul(jnp.transpose(S[i]), fvp[i]))
        if parent[i] != 0:
            fvp[parent[i] - 1] = fvp[parent[i] - 1] + jnp.matmul(jnp.transpose(Xup[i]), fvp[i])
    tau = jnp.reshape(jnp.array(tau), (NB, 1))

    return tau 




