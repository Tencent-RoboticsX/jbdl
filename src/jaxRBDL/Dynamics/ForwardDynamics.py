import numpy as np
import jax.numpy as jnp
from jaxRBDL.Model.JointModel import JointModel
from jaxRBDL.Math.CrossMotionSpace import CrossMotionSpace
from jaxRBDL.Math.CrossForceSpace import CrossForceSpace

def ForwardDynamics(model, q, qdot, tau):    
    q = q.flatten()
    qdot = qdot.flatten()
    tau = tau.flatten()
    a_grav = model["a_grav"].reshape(6, 1)
    NB = int(model["NB"])
    jtype = model["jtype"].flatten()
    jaxis = model['jaxis']
    parent = model['parent'].flatten().astype(int)
    try:
        Xtree = np.squeeze(model['Xtree'], axis=0)
        IA = np.squeeze(model['I'], axis=0).copy()
    except:
        Xtree = model['Xtree']
        IA = model['I'].copy()
    
    S = []
    Xup = []
    v = []
    c = []
    pA = []

    for i in range(NB):
        XJ, Si = JointModel(jtype[i], jaxis[i], q[i])
        S.append(Si)
        vJ = jnp.multiply(S[i], qdot[i])
        Xup.append(jnp.matmul(XJ, Xtree[i]))
        if parent[i] == 0:
            v.append(vJ)
            c.append(jnp.zeros((6, 1)))
        else:
            v.append(jnp.add(jnp.matmul(Xup[i], v[parent[i]-1]), vJ))
            c.append(jnp.matmul(CrossMotionSpace(v[i]), vJ))
        pA.append(jnp.matmul(CrossForceSpace(v[i]), jnp.matmul(IA[i], v[i])))



    U = [jnp.empty((0,))] * NB
    d = [jnp.empty((0,))] * NB
    u = [jnp.empty((0,))] * NB 

    for i in range(NB-1, -1, -1):
        U[i] = jnp.matmul(IA[i], S[i])
        d[i] = jnp.squeeze(jnp.matmul(S[i].transpose(), U[i]))
        u[i] = tau[i] - jnp.squeeze(jnp.matmul(S[i].transpose(), pA[i]))
        if parent[i] != 0:
            Ia = IA[i] - jnp.matmul(U[i]/ d[i], jnp.transpose(U[i]))
            pa = pA[i] + jnp.matmul(Ia, c[i]) + jnp.multiply(U[i], u[i]) / d[i]
            IA[parent[i] - 1] = IA[parent[i] - 1] + jnp.matmul(jnp.matmul(jnp.transpose(Xup[i]), Ia), Xup[i])
            pA[parent[i] - 1] = pA[parent[i] - 1] + jnp.matmul(jnp.transpose(Xup[i]), pa)


    a = []
    qddot = []

    for i in range(NB):
        if parent[i] == 0:
            a.append(jnp.matmul(Xup[i],-a_grav) + c[i])
        else:
            a.append(jnp.matmul(Xup[i], a[parent[i] - 1]) + c[i])
        qddot.append((u[i] - jnp.squeeze(jnp.matmul(jnp.transpose(U[i]), a[i])))/d[i])
        
        a[i] = a[i] + jnp.multiply(S[i],  qddot[i])

    qddot = jnp.reshape(jnp.stack(qddot), (NB, 1))
    return qddot





            

