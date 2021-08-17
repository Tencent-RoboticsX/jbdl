import numpy as np
from jbdl.rbdl.kinematics import calc_point_acceleration, calc_point_acceleration_core
import jax.numpy as jnp
from jax.api import jit
from functools import partial
from jbdl.rbdl.utils import xyz2int
from jax import lax

@partial(jit, static_argnums=(4, 6, 7, 8, 9, 10, 11))
def calc_contact_jdot_qdot_core_jit_flag(x_tree, q, qdot, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf):
    JdotQdot = []
    fbool_contact = jnp.heaviside(flag_contact, 0.0)
    qddot = jnp.zeros((nb,))
    for i in range(nc):
        JdotQdoti = jnp.empty((0, 1))
        # print(fbool_contact.shape)
        # print(fbool_contact[i].shape)
        JdQd = fbool_contact[i] * calc_point_acceleration_core(x_tree, parent, jtype, jaxis, idcontact[i], q, qdot, qddot, contactpoint[i])
    
        if nf == 2:
            JdotQdoti = JdQd[[0, 2], :] # only x\z direction
        elif nf == 3:
            JdotQdoti = JdQd
   

        JdotQdot.append(JdotQdoti)

    JdotQdot = jnp.concatenate(JdotQdot, axis=0)
    return JdotQdot


@partial(jit, static_argnums=(4, 6, 7, 8, 9, 10, 11))
def calc_contact_jdot_qdot_extend_core(x_tree, q, qdot, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf):
    JdotQdot = []
    # fbool_contact = jnp.heaviside(flag_contact, 0.0)
    qddot = jnp.zeros((nb,))
    for i in range(nc):
        JdotQdoti = jnp.empty((0, 1))

        JdQd = lax.cond(
            flag_contact[i],
            lambda _: calc_point_acceleration_core(x_tree, parent, jtype, jaxis, idcontact[i], q, qdot, qddot, contactpoint[i]),
            lambda _: jnp.zeros((3,1)),
            None
        )

    
        if nf == 2:
            JdotQdoti = JdQd[[0, 2], :] # only x\z direction
        elif nf == 3:
            JdotQdoti = JdQd
   

        JdotQdot.append(JdotQdoti)

    JdotQdot = jnp.concatenate(JdotQdot, axis=0)
    return JdotQdot


# @partial(jit, static_argnums=(4, 5, 6, 7, 8, 9, 10, 11))
def calc_contact_jdot_qdot_core(x_tree, q, qdot, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf):
    JdotQdot = []
    qddot = jnp.zeros((nb,))
    for i in range(nc):
        JdotQdoti = jnp.empty((0, 1))
        if flag_contact[i] != 0.0:
            # print(jaxis)
            JdQd = calc_point_acceleration_core(x_tree, parent, jtype, jaxis, idcontact[i], q, qdot, qddot, contactpoint[i])
     
            if nf == 2:
                JdotQdoti = JdQd[[0, 2], :] # only x\z direction
            elif nf == 3:
                JdotQdoti = JdQd
   

        JdotQdot.append(JdotQdoti)
    JdotQdot = jnp.concatenate(JdotQdot, axis=0)
    return JdotQdot

# def calc_contact_jdot_qdot(model: dict, q: np.ndarray, qdot: np.ndarray, flag_contact: np.ndarray)->np.ndarray:
#     nc = int(model["nc"])
#     nb = int(model["nb"])
#     nf = int(model["nf"])
#     q = q.flatten()
#     qdot = qdot.flatten()
#     flag_contact = flag_contact.flatten()
#     idcontact = model["idcontact"]
#     contactpoint = model["contactpoint"]

    
#     JdotQdot = []
#     for i in range(nc):
#         JdotQdoti = np.empty((0, 1))
#         if flag_contact[i] != 0.0:
#             JdQd = calc_point_acceleration(model, q, qdot, np.zeros((nb, 1)), idcontact[i], contactpoint[i])
#             if nf == 2:
#                 JdotQdoti = JdQd[[0, 2], :] # only x\z direction
#             elif nf == 3:
#                 JdotQdoti = JdQd
   

#         JdotQdot.append(JdotQdoti)

#     JdotQdot = np.asfarray(np.concatenate(JdotQdot, axis=0))

#     return JdotQdot

def calc_contact_jdot_qdot(model: dict, q: np.ndarray, qdot: np.ndarray, flag_contact: np.ndarray)->np.ndarray:
    nc = int(model["nc"])
    nb = int(model["nb"])
    nf = int(model["nf"])
    x_tree = model["x_tree"]
    contactpoint = model["contactpoint"],
    idcontact = tuple(model["idcontact"])
    parent = tuple(model["parent"])
    jtype = tuple(model["jtype"])
    jaxis = xyz2int(model["jaxis"])
    contactpoint = model["contactpoint"]
    flag_contact = flag_contact
    JdotQdot = calc_contact_jdot_qdot_core(x_tree, q, qdot, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf)

    return JdotQdot
                
