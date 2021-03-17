import numpy as np
from jaxRBDL.Kinematics.CalcPointJacobian import CalcPointJacobian

def CalcContactJacobian(model: dict, q: np.ndarray, flag_contact: np.ndarray, nf: int=3)->np.ndarray:
    NC = int(model["NC"])
    NB = int(model["NB"])
    q = q.flatten()
    flag_contact = flag_contact.flatten()

    try: 
        idcontact = np.squeeze(model["idcontact"], axis=0).astype(int)
        contactpoint = np.squeeze(model["contactpoint"], axis=0)
    except:
        idcontact = model["idcontact"]
        contactpoint = model["contactpoint"]

    Jc = []
    for i in range(NC):
        Jci = np.empty((0, NB))
        if flag_contact[i] != 0.0:
            # Calculate Jacobian
            J = CalcPointJacobian(model, q, idcontact[i], contactpoint[i])

            # Make Jacobian full rank according to contact model
            if nf == 2:
                Jci = J[[0, 2], :] # only x\z direction
            elif nf == 3:
                Jci = J          
        Jc.append(Jci)

    Jc = np.asfarray(np.concatenate(Jc, axis=0))
    return Jc
