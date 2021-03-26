from typing import Tuple
import numpy as np
from jaxRBDL.Contact.CalcContactJacobian import CalcContactJacobian
from jaxRBDL.Contact.CalcContactJdotQdot import CalcContactJdotQdot
from jaxRBDL.Contact.CalcContactForcePD import CalcContactForcePD
from jaxRBDL.Contact.GetContactForce import GetContactForce
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import gmres

def CheckContactForce(model: dict, flag_contact: np.ndarray, fqp: np.ndarray, nf: int):
    NC = int(model["NC"])
    flag_contact = flag_contact.flatten()

    flag_recalc = 0
    flag_newcontact = flag_contact

    k = 0
    for i in range(NC):
        if flag_contact[i] != 0:
            if fqp[k*nf+nf-1, 0] < 0:
                flag_newcontact[i] = 0
                flag_recalc = 1
                break
            k = k+1

    return flag_newcontact, flag_recalc

def CalcContactForceDirect(model: dict, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray, flag_contact: np.ndarray, nf: int):
    NB = int(model["NB"])
    NC = int(model["NC"])
        
    flag_recalc = 1
    fqp = np.empty((0, 1))
    flcp = np.empty((0, 1))  
    while flag_recalc:
        if np.sum(flag_contact)==0:
            fqp = np.zeros((NC*nf, 1))
            flcp = np.zeros((NB, 1))
            fc = np.zeros((3*NC,))
            fcqp = np.zeros((3*NC,))
            fcpd = np.zeros((3*NC,))
            break

        # Calculate contact force
        Jc = CalcContactJacobian(model, q, flag_contact, nf)
        JcdotQdot = CalcContactJdotQdot(model, q, qdot, flag_contact, nf)

        M = np.matmul(np.matmul(Jc, model["Hinv"]), np.transpose(Jc))
        # print(M)
        tau = tau.reshape(-1, 1)
        d0 = np.matmul(np.matmul(Jc, model["Hinv"]), tau - model["C"])
        d = np.add(d0, JcdotQdot )
        
        #TODO M may be sigular for nf=3 
        fqp = -np.linalg.solve(M,d)
     

        # Check whether the Fz is positive
        flag_contact, flag_recalc = CheckContactForce(model, flag_contact, fqp, nf)
        if flag_recalc == 0:
            flcp = np.matmul(np.transpose(Jc), fqp)
        
        contact_force_kp = np.array([10000.0, 10000.0, 10000.0])
        contact_force_kd = np.array([1000.0, 1000.0, 1000.0])

        # Calculate contact force from PD controller
        fpd = CalcContactForcePD(model, q, qdot, flag_contact, contact_force_kp, contact_force_kd, nf)
        fc, fcqp, fcpd = GetContactForce(model, fqp, fpd, flag_contact, nf)  


    return flcp, fqp, fc, fcqp, fcpd