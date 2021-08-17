from functools import partial
import numpy as np
from jbdl.rbdl.kinematics import calc_point_jacobian_core
import jax.numpy as jnp
from jax.api import jit
from jax import lax
from jbdl.rbdl.utils import xyz2int


@partial(jit, static_argnums=(3, 5, 6, 7, 8, 9, 10))
def calc_contact_jacobian_core_jit_flag(
    x_tree, q, contactpoint, idcontact, flag_contact,
    parent, jtype, jaxis, nb, nc, nf):
    jc = []
    fbool_contact = jnp.heaviside(flag_contact, 0.0)
    for i in range(nc):
        jci = jnp.empty((0, nb))
   
        # Calculate Jacobian
        jac = fbool_contact[i] * calc_point_jacobian_core(x_tree, parent, jtype, jaxis, nb, idcontact[i], q, contactpoint[i])

        # Make Jacobian full rank according to contact model
        if nf == 2:
            jci = jac[[0, 2], :] # only x\z direction
        elif nf == 3:
            jci = jac          
        jc.append(jci)
    jc = jnp.concatenate(jc, axis=0)
    return jc

@partial(jit, static_argnums=(3, 5, 6, 7, 8, 9, 10))
def calc_contact_jacobian_extend_core(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf):
    jc = []
    # fbool_contact = jnp.heaviside(flag_contact, 0.0)
    for i in range(nc):
        jci = jnp.empty((0, nb))
   
        # Calculate Jacobian
        jac = lax.cond(
            flag_contact[i], 
            lambda _: calc_point_jacobian_core(x_tree, parent, jtype, jaxis, nb, idcontact[i], q, contactpoint[i]),
            lambda _: jnp.zeros((3, nb)),
            None
        )
        # J = fbool_contact[i] * calc_point_jacobian_core(x_tree, parent, jtype, jaxis, nb, idcontact[i], q, contactpoint[i])

        # Make Jacobian full rank according to contact model
        if nf == 2:
            jci = jac[[0, 2], :] # only x\z direction
        elif nf == 3:
            jci = jac          
        jc.append(jci)
    jc = jnp.concatenate(jc, axis=0)
    return jc


# @partial(jit, static_argnums=(3, 4, 5, 6, 7, 8, 9, 10))
def calc_contact_jacobian_core(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf):
    jc = []
    for i in range(nc):
        jci = jnp.empty((0, nb))
        if flag_contact[i] != 0.0:
            # Calculate Jacobian
            jac = calc_point_jacobian_core(x_tree, parent, jtype, jaxis, nb, idcontact[i], q, contactpoint[i])

            # Make Jacobian full rank according to contact model
            if nf == 2:
                jci = jac[[0, 2], :] # only x\z direction
            elif nf == 3:
                jci = jac          
        jc.append(jci)
    jc = jnp.concatenate(jc, axis=0)
    return jc

def calc_contact_jacobian(model: dict, q: np.ndarray, flag_contact: np.ndarray)->np.ndarray:
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
    jc = calc_contact_jacobian_core(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf)
    return jc
