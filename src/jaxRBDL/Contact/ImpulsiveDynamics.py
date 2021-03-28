import numpy as np
from numpy.core.shape_base import hstack
from jaxRBDL.Contact.CalcContactJacobian import CalcContactJacobian
from numpy.linalg import matrix_rank


def ImpulsiveDynamics(model: dict, q: np.ndarray, qdot: np.ndarray, flag_contact:np.ndarray)->np.ndarray:
    q = q.flatten()
    qdot = qdot.flatten()
    NB = int(model["NB"])
    nf = model["nf"]

    try: 
        H = np.squeeze(model["H"], axis=0)
    except:
        H = model["H"]


    Jc = CalcContactJacobian(model, q, flag_contact)
    rankJc = matrix_rank(Jc)

    # Calcualet implusive dynamics for qdot after impulsive
    A0 = np.hstack([H, -np.transpose(Jc)])
    A1 = np.hstack([Jc, np.zeros((rankJc, rankJc))])
    A = np.vstack([A0, A1])
    b0 = np.matmul(H, qdot)
    b1 = np.zeros((rankJc, ))
    b = np.hstack([b0, b1])
    QdotI = np.linalg.solve(A, b)
    qdot_impulse = QdotI[0:NB]
    qdot_impulse = qdot_impulse.reshape(-1, 1)
    return qdot_impulse
 
