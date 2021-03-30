from functools import partial
import numpy as np
from jaxRBDL.Kinematics.CalcPointJacobian import CalcPointJacobian, CalcPointJacobianCore
import jax.numpy as jnp
from jax.api import jit

@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8, 9, 10))
def CalcContactJacobianCore(Xtree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf):
    Jc = []
    for i in range(NC):
        Jci = jnp.empty((0, NB))
        if flag_contact[i] != 0.0:
            # Calculate Jacobian
            J = CalcPointJacobianCore(Xtree, parent, jtype, jaxis, NB, idcontact[i], q, contactpoint[i])

            # Make Jacobian full rank according to contact model
            if nf == 2:
                Jci = J[[0, 2], :] # only x\z direction
            elif nf == 3:
                Jci = J          
        Jc.append(Jci)
    Jc = jnp.concatenate(Jc, axis=0)
    return Jc

def CalcContactJacobian(model: dict, q: np.ndarray, flag_contact: np.ndarray)->np.ndarray:
    NC = int(model["NC"])
    NB = int(model["NB"])
    nf = int(model["nf"])
    Xtree = model["Xtree"]
    contactpoint = model["contactpoint"],
    idcontact = tuple(model["idcontact"])
    parent = tuple(model["parent"])
    jtype = tuple(model["jtype"])
    jaxis = model["jaxis"]
    contactpoint = model["contactpoint"]
    flag_contact = flag_contact
    Jc = CalcContactJacobianCore(Xtree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
    return Jc
    

# def CalcContactJacobian(model: dict, q: np.ndarray, flag_contact: np.ndarray)->np.ndarray:
#     NC = int(model["NC"])
#     NB = int(model["NB"])
#     nf = int(model["nf"])
#     q = q.flatten()
#     flag_contact = flag_contact.flatten()
#     idcontact = model["idcontact"]
#     contactpoint = model["contactpoint"]

#     Jc = []
#     for i in range(NC):
#         Jci = np.empty((0, NB))
#         if flag_contact[i] != 0.0:
#             # Calculate Jacobian
#             J = CalcPointJacobian(model, q, idcontact[i], contactpoint[i])

#             # Make Jacobian full rank according to contact model
#             if nf == 2:
#                 Jci = J[[0, 2], :] # only x\z direction
#             elif nf == 3:
#                 Jci = J          
#         Jc.append(Jci)

#     Jc = np.asfarray(np.concatenate(Jc, axis=0))
#     return Jc
