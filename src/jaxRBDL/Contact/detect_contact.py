import numpy as np
from jaxRBDL.kinematics import calc_body_to_base_coordinates, calc_body_to_base_coordinates_core
from jaxRBDL.kinematics import calc_point_velocity, calc_point_velocity_core
import jax.numpy as jnp
from functools import partial
from jax.api import jit

# @partial(jit, static_argnums=(2, 3, 4))
# def DeterminContactTypeCore(pos, vel, contact_pos_lb, contact_vel_lb, contact_vel_ub):
#     pos = jnp.reshape(pos, (-1,))
#     vel = jnp.reshape(vel, (-1,))
#     if pos[2] < contact_pos_lb[2]:
#         if vel[2] < contact_vel_lb[2]:
#             contact_type = 2 # impact    
#         elif vel[2] > contact_vel_ub[2]:
#             contact_type = 0 # uncontact
#         else:
#             contact_type = 1 # contact 
#     else:
#         contact_type = 0  # uncontact
#     return contact_type

@jit
def determin_contact_type_core(pos, vel, contact_pos_lb, contact_vel_lb, contact_vel_ub):
    pos_z = pos[2, :]
    vel_z = vel[2, :]
    contact_pos_lb_z = contact_pos_lb[2]
    contact_vel_lb_z = contact_vel_lb[2]
    contact_vel_ub_z = contact_vel_ub[2]

    # contact_type = jnp.maximum(jnp.sign(contact_pos_lb_z - pos_z), 0.0) \
    #     * jnp.maximum(jnp.sign(contact_vel_ub_z - vel_z), 0.0)
    # contact_type = contact_type * (jnp.maximum(jnp.sign(contact_vel_lb_z - vel_z), 0) + 1.0)

    contact_type = jnp.heaviside(contact_pos_lb_z - pos_z, 0.0) \
        * jnp.heaviside(contact_vel_ub_z - vel_z, 0.0)
    contact_type = contact_type * (jnp.heaviside(contact_vel_lb_z - vel_z, 0) + 1.0)   


    return contact_type
    

def determin_contact_type(pos: np.ndarray, vel: np.ndarray, contact_cond: dict)->int:
    pos = pos.flatten()
    vel = vel.flatten()
    contact_pos_lb = contact_cond["contact_pos_lb"].flatten()
    contact_vel_lb = contact_cond["contact_vel_lb"].flatten()
    contact_vel_ub = contact_cond["contact_vel_ub"].flatten()

    if pos[2] < contact_pos_lb[2]:
        if vel[2] < contact_vel_lb[2]:
            contact_type = 2 # impact    
        elif vel[2] > contact_vel_ub[2]:
            contact_type = 0 # uncontact
        else:
            contact_type = 1 # contact 
    else:
        contact_type = 0  # uncontact

    return contact_type

@partial(jit, static_argnums=(7, 8, 9, 10, 11))
def detect_contact_core(Xtree, q, qdot, contactpoint, contact_pos_lb, contact_vel_lb, contact_vel_ub,\
    idcontact, parent, jtype, jaxis, NC):
    # flag_contact_list = []
    end_pos_list = []
    end_vel_list = []
    for i in range(NC):
        # Calcualte pos and vel of foot endpoint, column vector
        endpos_item = calc_body_to_base_coordinates_core(Xtree, parent, jtype, jaxis, idcontact[i], q, contactpoint[i])
        endvel_item = calc_point_velocity_core(Xtree, parent, jtype, jaxis, idcontact[i], q, qdot, contactpoint[i])
        end_pos_list.append(endpos_item)
        end_vel_list.append(endvel_item)

    end_pos = jnp.hstack(end_pos_list)
    end_vel = jnp.hstack(end_vel_list)

    flag_contact = determin_contact_type_core(end_pos, end_vel, contact_pos_lb, contact_vel_lb, contact_vel_ub)
    
    return flag_contact
        
    
def detect_contact(model: dict, q: np.ndarray, qdot: np.ndarray)->np.ndarray:
    contact_cond = model["contact_cond"]
    NC = int(model["NC"])
    Xtree = model["Xtree"]
    contactpoint = model["contactpoint"],
    idcontact = tuple(model["idcontact"])
    parent = tuple(model["parent"])
    jtype = tuple(model["jtype"])
    jaxis = model["jaxis"]
    contactpoint = model["contactpoint"]
    contact_pos_lb = contact_cond["contact_pos_lb"]
    contact_vel_lb = contact_cond["contact_vel_lb"]
    contact_vel_ub = contact_cond["contact_vel_ub"]
    flag_contact = detect_contact_core(Xtree, q, qdot, contactpoint, contact_pos_lb, contact_vel_lb, contact_vel_ub, idcontact, parent, jtype, jaxis, NC)
    # flag_contact = DeterminContactTypeCore(end_pos, end_vel, )

    return tuple(flag_contact)


def detect_contact_v0(model: dict, q: np.ndarray, qdot: np.ndarray)->np.ndarray:
    NC = int(model["NC"])
    contact_cond = model["contact_cond"]
    idcontact = model["idcontact"]
    contactpoint = model["contactpoint"]

    flag_contact = np.zeros((NC, 1))


    flag_contact_list = []

    for i in range(NC):
        # Calcualte pos and vel of foot endpoint, column vector
        endpos_item = calc_body_to_base_coordinates(model, q, idcontact[i], contactpoint[i])
        endvel_item = calc_point_velocity(model, q, qdot, idcontact[i], contactpoint[i])

        # Detect contact
        flag_contact_list.append(determin_contact_type(endpos_item, endvel_item, contact_cond))

    flag_contact = np.asfarray(flag_contact_list).flatten()

    return tuple(flag_contact)
    
if __name__ == "__main__":
    pass

