import numpy as np
from jaxRBDL.Kinematics.CalcPointAcceleraion import CalcPointAcceleration, CalcPointAccelerationCore
import jax.numpy as jnp
from jax.api import jit
from functools import partial

@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9, 10, 11))
def CalcContactJdotQdotCore(Xtree, q, qdot, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf):
    JdotQdot = []
    qddot = jnp.zeros((NB,))
    for i in range(NC):
        JdotQdoti = jnp.empty((0, 1))
        if flag_contact[i] != 0.0:
            JdQd = CalcPointAccelerationCore(Xtree, parent, jtype, jaxis, idcontact[i], q, qdot, qddot, contactpoint[i])
     
            if nf == 2:
                JdotQdoti = JdQd[[0, 2], :] # only x\z direction
            elif nf == 3:
                JdotQdoti = JdQd
   

        JdotQdot.append(JdotQdoti)

    JdotQdot = jnp.concatenate(JdotQdot, axis=0)
    return JdotQdot

# def CalcContactJdotQdot(model: dict, q: np.ndarray, qdot: np.ndarray, flag_contact: np.ndarray)->np.ndarray:
#     NC = int(model["NC"])
#     NB = int(model["NB"])
#     nf = int(model["nf"])
#     q = q.flatten()
#     qdot = qdot.flatten()
#     flag_contact = flag_contact.flatten()
#     idcontact = model["idcontact"]
#     contactpoint = model["contactpoint"]

    
#     JdotQdot = []
#     for i in range(NC):
#         JdotQdoti = np.empty((0, 1))
#         if flag_contact[i] != 0.0:
#             JdQd = CalcPointAcceleration(model, q, qdot, np.zeros((NB, 1)), idcontact[i], contactpoint[i])
#             if nf == 2:
#                 JdotQdoti = JdQd[[0, 2], :] # only x\z direction
#             elif nf == 3:
#                 JdotQdoti = JdQd
   

#         JdotQdot.append(JdotQdoti)

#     JdotQdot = np.asfarray(np.concatenate(JdotQdot, axis=0))

#     return JdotQdot

def CalcContactJdotQdot(model: dict, q: np.ndarray, qdot: np.ndarray, flag_contact: np.ndarray)->np.ndarray:
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

    JdotQdot = CalcContactJdotQdotCore(Xtree, q, qdot, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)

    return JdotQdot
                
