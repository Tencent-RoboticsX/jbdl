import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.lib.function_base import append
from numpy.lib.type_check import asfarray
from pyRBDL.Model.JointModel import JointModel
from pyRBDL.Math.CrossMotionSpace import CrossMotionSpace
from pyRBDL.Math.CrossForceSpace import CrossForceSpace

def ForwardDynamics(model: dict, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray)->np.ndarray:
    """ForwadrdDynamics  Forward Dynamics via Articulated-Body Algorithm.
    ForwardDynamics(model, q, qdot, tau) calculates the forward dynamics of a 
    kinematic tree via the articulated-body algorithm.  q, qdot and tau are 
    vectors of joint position, velocity and force variables; and the return
    value is a vector of joint acceleration variables. 

    Args:
        model (dict): dictionary of model specification
        q (np.ndarray): an array of joint position
        qdot (np.ndarray): an array of joint velocity
        tau (np.ndarray): an array of joint force

    Returns:
        np.ndarray: [description]
    """    
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
        vJ = S[i] * qdot[i]
        Xup.append(np.matmul(XJ, Xtree[i]))
        if parent[i] == 0:
            v.append(vJ)
            c.append(np.zeros((6, 1)))
        else:
            v.append(np.add(np.matmul(Xup[i], v[parent[i]-1]), vJ))
            c.append(np.matmul(CrossMotionSpace(v[i]), vJ))
        pA.append(np.matmul(CrossForceSpace(v[i]), np.matmul(IA[i], v[i])))



    U = [np.empty((0,))] * NB
    d = [np.empty((0,))] * NB
    u = [np.empty((0,))] * NB 

    for i in range(NB-1, -1, -1):
        U[i] = np.matmul(IA[i], S[i])
        d[i] = np.matmul(S[i].transpose(), U[i]).squeeze()
        u[i] = tau[i] - np.matmul(S[i].transpose(), pA[i]).squeeze()
        if parent[i] != 0:
            Ia = IA[i] - np.matmul(U[i]/ d[i], U[i].transpose())
            pa = pA[i] + np.matmul(Ia, c[i]) + U[i] * u[i] / d[i]
            IA[parent[i] - 1] = IA[parent[i] - 1] + np.matmul(np.matmul(Xup[i].transpose(), Ia), Xup[i])
            pA[parent[i] - 1] = pA[parent[i] - 1] + np.matmul(Xup[i].transpose(), pa)


    a = []
    qddot = []

    for i in range(NB):
        if parent[i] == 0:
            a.append(np.matmul(Xup[i],-a_grav) + c[i])
        else:
            a.append(np.matmul(Xup[i], a[parent[i] - 1]) + c[i])
        qddot.append((u[i] - np.matmul(U[i].transpose(), a[i]).squeeze())/d[i])
        
        a[i] = a[i] + S[i] * qddot[i]

    qddot = np.asfarray(qddot).reshape(NB, 1)

    return qddot





            

