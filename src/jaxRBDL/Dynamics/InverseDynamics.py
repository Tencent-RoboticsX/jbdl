import numpy as np
from pyRBDL.Model.JointModel import JointModel
from pyRBDL.Math.CrossMotionSpace import CrossMotionSpace
from pyRBDL.Math.CrossForceSpace import CrossForceSpace

def InverseDynamics(model: dict, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray)-> np.ndarray:
    
    a_grav = model["a_grav"].reshape(6, 1)
    qdot = qdot.flatten()
    qddot = qddot.flatten()
    NB = int(model["NB"])
    tau = np.zeros((NB,1))
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
        vJ = S[i] * qdot[i]
        Xup.append(np.matmul(XJ, Xtree[i]))
        if parent[i] == 0:
            v.append(vJ)
            avp.append(np.matmul(Xup[i], -a_grav))
        else:
            v.append(np.matmul(Xup[i], v[parent[i] - 1]) + vJ)
            avp.append(np.matmul(Xup[i], avp[parent[i] - 1]) + S[i] * qddot[i] + np.matmul(CrossMotionSpace(v[i]), vJ))
        fvp.append(np.matmul(I[i], avp[i]) + np.matmul(np.matmul(CrossForceSpace(v[i]), I[i]), v[i]))

    for i in range(NB-1, -1, -1):
        tau[i] = np.matmul(S[i].transpose(), fvp[i]).squeeze()
        if parent[i] != 0:
            fvp[parent[i] - 1] = fvp[parent[i] - 1] + np.matmul(Xup[i].transpose(), fvp[i])

    return tau 




