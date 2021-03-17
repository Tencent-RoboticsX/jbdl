import numpy as np
from jaxRBDL.Kinematics.CalcPointAcceleraion import CalcPointAcceleration

def CalcContactJdotQdot(model: dict, q: np.ndarray, qdot: np.ndarray, flag_contact: np.ndarray, nf: int=3)->np.ndarray:
    NC = int(model["NC"])
    NB = int(model["NB"])
    q = q.flatten()
    qdot = qdot.flatten()
    flag_contact = flag_contact.flatten()

    try: 
        idcontact = np.squeeze(model["idcontact"], axis=0).astype(int)
        contactpoint = np.squeeze(model["contactpoint"], axis=0)
    except:
        idcontact = model["idcontact"]
        contactpoint = model["contactpoint"]

    
    JdotQdot = []
    for i in range(NC):
        JdotQdoti = np.empty((0, 1))
        if flag_contact[i] != 0.0:
            JdQd = CalcPointAcceleration(model, q, qdot, np.zeros((NB, 1)), idcontact[i], contactpoint[i])
            if nf == 2:
                JdotQdoti = JdQd[[0, 2], :] # only x\z direction
            elif nf == 3:
                JdotQdoti = JdQd
   

        JdotQdot.append(JdotQdoti)

    JdotQdot = np.asfarray(np.concatenate(JdotQdot, axis=0))

    return JdotQdot
                
