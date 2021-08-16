from functools import partial
import numpy as np
from jbdl.rbdl.kinematics import calc_point_jacobian_core
import jax.numpy as jnp
from jax.api import jit
from jax import lax
from jbdl.rbdl.utils import xyz2int
from jax.tree_util import tree_flatten

@partial(jit, static_argnums=(3, 5, 6, 7, 8, 9, 10))
def calc_contact_jacobian_core_jit_flag(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf):
    # fbool_contact = jnp.heaviside(flag_contact, 0.0)
    # idcontact = jnp.array(idcontact, dtype=int)
    # contactpoint = jnp.vstack(contactpoint)

    # carry = (x_tree, q)
    # xs = (fbool_contact, idcontact, contactpoint)

    # def f(carry, xs):
    #     x_tree, q = carry

    #     fbool, body_id, point_pos = xs
    #     # body_id is not static
    #     ys = fbool * calc_point_jacobian_core(x_tree, parent, jtype, jaxis, NB, body_id, q, point_pos)
    #     return carry, ys

    # J = lax.scan(f, carry, xs)
    
    # return J

    Jc = []
    fbool_contact = jnp.heaviside(flag_contact, 0.0)
    for i in range(NC):
        Jci = jnp.empty((0, NB))
   
        # Calculate Jacobian
        J = fbool_contact[i] * calc_point_jacobian_core(x_tree, parent, jtype, jaxis, NB, idcontact[i], q, contactpoint[i])

        # Make Jacobian full rank according to contact model
        if nf == 2:
            Jci = J[[0, 2], :] # only x\z direction
        elif nf == 3:
            Jci = J          
        Jc.append(Jci)
    Jc = jnp.concatenate(Jc, axis=0)
    return Jc

@partial(jit, static_argnums=(3, 5, 6, 7, 8, 9, 10))
def calc_contact_jacobian_extend_core(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf):
    Jc = []
    # fbool_contact = jnp.heaviside(flag_contact, 0.0)
    for i in range(NC):
        Jci = jnp.empty((0, NB))
   
        # Calculate Jacobian
        J = lax.cond(
            flag_contact[i], 
            lambda _: calc_point_jacobian_core(x_tree, parent, jtype, jaxis, NB, idcontact[i], q, contactpoint[i]),
            lambda _: jnp.zeros((3, NB)),
            None
        )
        # J = fbool_contact[i] * calc_point_jacobian_core(x_tree, parent, jtype, jaxis, NB, idcontact[i], q, contactpoint[i])

        # Make Jacobian full rank according to contact model
        if nf == 2:
            Jci = J[[0, 2], :] # only x\z direction
        elif nf == 3:
            Jci = J          
        Jc.append(Jci)
    Jc = jnp.concatenate(Jc, axis=0)
    return Jc
    # for idcontact_elem in idcontact:
    #     flag_contact[i]
    # def f(carry, x):
    #     flag,  contactpoint_elem = x
    #     new_x = lax.cond(
    #         flag,
    #         lambda _: calc_point_jacobian_core(x_tree, parent, jtype, jaxis, NB, 9, q, contactpoint_elem),
    #         lambda _: jnp.zeros((3, NB)),
    #         None
    #     )
    #     carry += 1
    #     return carry, new_x 

    

    # _, seq_Jc = lax.scan(f, 0, (jnp.array(flag_contact),  jnp.array(contactpoint)))
    # Jc = []
    # for i in range(NC):
    #     Jci = jnp.empty((0, NB))
    #     if flag_contact[i] != 0.0:
    #         # Calculate Jacobian
    #         J = calc_point_jacobian_core(x_tree, parent, jtype, jaxis, NB, idcontact[i], q, contactpoint[i])

    #         # Make Jacobian full rank according to contact model
    #         if nf == 2:
    #             Jci = J[[0, 2], :] # only x\z direction
    #         elif nf == 3:
    #             Jci = J          
    #     Jc.append(Jci)
    # Jc = jnp.concatenate(Jc, axis=0)
    # return seq_Jc

# @partial(jit, static_argnums=(3, 4, 5, 6, 7, 8, 9, 10))
def calc_contact_jacobian_core(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf):
    Jc = []
    for i in range(NC):
        Jci = jnp.empty((0, NB))
        if flag_contact[i] != 0.0:
            # Calculate Jacobian
            J = calc_point_jacobian_core(x_tree, parent, jtype, jaxis, NB, idcontact[i], q, contactpoint[i])

            # Make Jacobian full rank according to contact model
            if nf == 2:
                Jci = J[[0, 2], :] # only x\z direction
            elif nf == 3:
                Jci = J          
        Jc.append(Jci)
    Jc = jnp.concatenate(Jc, axis=0)
    return Jc

def calc_contact_jacobian(model: dict, q: np.ndarray, flag_contact: np.ndarray)->np.ndarray:
    NC = int(model["NC"])
    NB = int(model["NB"])
    nf = int(model["nf"])
    x_tree = model["x_tree"]
    contactpoint = model["contactpoint"],
    idcontact = tuple(model["idcontact"])
    parent = tuple(model["parent"])
    jtype = tuple(model["jtype"])
    jaxis = xyz2int(model["jaxis"])
    contactpoint = model["contactpoint"]
    flag_contact = flag_contact
    Jc = calc_contact_jacobian_core(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
    return Jc
    

# def calc_contact_jacobian(model: dict, q: np.ndarray, flag_contact: np.ndarray)->np.ndarray:
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
#             J = calc_point_jacobian(model, q, idcontact[i], contactpoint[i])

#             # Make Jacobian full rank according to contact model
#             if nf == 2:
#                 Jci = J[[0, 2], :] # only x\z direction
#             elif nf == 3:
#                 Jci = J          
#         Jc.append(Jci)

#     Jc = np.asfarray(np.concatenate(Jc, axis=0))
#     return Jc

# if __name__ == "__main__":
#     flag_contact = 
