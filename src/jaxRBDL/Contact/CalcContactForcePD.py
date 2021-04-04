import numpy as np
from jaxRBDL.Kinematics import calc_body_to_base_coordinates
from jaxRBDL.Kinematics.CalcPointVelocity import CalcPointVelocity


def CalcContactForcePD(model: dict, q: np.ndarray, qdot: np.ndarray, 
                       flag_contact: np.ndarray)->np.ndarray:
    
    NC = int(model["NC"])
    NB = int(model["NB"])
    nf = int(model["nf"])
    q = q.flatten()
    qdot = qdot.flatten()
    flag_contact = flag_contact
    contact_cond = model["contact_cond"]
    contact_force_kp = contact_cond["contact_force_kp"].flatten()
    contact_force_kd = contact_cond["contact_force_kd"].flatten()

    try: 
        idcontact = np.squeeze(model["idcontact"], axis=0).astype(int)
        contactpoint = np.squeeze(model["contactpoint"], axis=0)
    except:
        idcontact = model["idcontact"]
        contactpoint = model["contactpoint"]



    if np.all(flag_contact==0):
        fpd = np.zeros((NC*nf, 1))

    else:
        endpos = np.zeros((3, NC))
        endvel = np.zeros((3, NC))
        fpd = []

        for i in range(NC):
            if flag_contact[i] != 0:
                # Calcualte pos and vel of foot endpoint
                endpos[:, i:i+1] = calc_body_to_base_coordinates(model, q, idcontact[i], contactpoint[i])
                endvel[:, i:i+1] = CalcPointVelocity(model, q, qdot, idcontact[i], contactpoint[i])
                
                # Calculate contact force by PD controller

                if nf==2:
                    fpdi = np.zeros((2, 1))
                    fpdi[0, 0] = -contact_force_kp[1]*endvel[0, i]
                    fpdi[1, 0] = -contact_force_kp[2]*endpos[2, i] - contact_force_kd[2] * min(endvel[2, i], 0.0)
                elif nf==3:
                    fpdi = np.zeros((3, 1))
                    fpdi[0, 0] = -contact_force_kp[0] * endvel[0, i]
                    fpdi[1, 0] = -contact_force_kp[1] * endvel[1, i]
                    fpdi[2, 0] = -contact_force_kp[2] * min(endpos[2, i], 0.0) - contact_force_kd[2] * min(endvel[2, i], 0.0)
                else:
                    fpdi = np.empty((0, 1))
             
                fpd.append(fpdi)
        fpd = np.asfarray(np.concatenate(fpd, axis=0))

    return fpd
    
