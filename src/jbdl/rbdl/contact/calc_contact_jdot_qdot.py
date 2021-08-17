from functools import partial
import numpy as np
from jbdl.rbdl.kinematics import calc_point_acceleration_core
import jax.numpy as jnp
from jax.api import jit
from jbdl.rbdl.utils import xyz2int
from jax import lax


@partial(jit, static_argnums=(4, 6, 7, 8, 9, 10, 11))
def calc_contact_jdot_qdot_core_jit_flag(
    x_tree, q, qdot, contactpoint, idcontact, flag_contact,
    parent, jtype, jaxis, nb, nc, nf):
    jdot_qdot = []
    fbool_contact = jnp.heaviside(flag_contact, 0.0)
    qddot = jnp.zeros((nb,))
    for i in range(nc):
        jdot_qdoti = jnp.empty((0, 1))
        jd_qd = fbool_contact[i] * \
            calc_point_acceleration_core(
                x_tree, parent, jtype, jaxis, idcontact[i], q, qdot, qddot, contactpoint[i])

        if nf == 2:
            jdot_qdoti = jd_qd[[0, 2], :]  # only x\z direction
        elif nf == 3:
            jdot_qdoti = jd_qd

        jdot_qdot.append(jdot_qdoti)

    jdot_qdot = jnp.concatenate(jdot_qdot, axis=0)
    return jdot_qdot


@partial(jit, static_argnums=(4, 6, 7, 8, 9, 10, 11))
def calc_contact_jdot_qdot_extend_core(
    x_tree, q, qdot, contactpoint, idcontact, flag_contact,
    parent, jtype, jaxis, nb, nc, nf):

    jdot_qdot = []
    qddot = jnp.zeros((nb,))

    for i in range(nc):
        jdot_qdoti = jnp.empty((0, 1))
        jd_qd = lax.cond(
            flag_contact[i],
            lambda _: calc_point_acceleration_core(x_tree, parent, jtype, jaxis, idcontact[i],
                q, qdot, qddot, contactpoint[i]),
            lambda _: jnp.zeros((3, 1)),
            None
        )

        if nf == 2:
            jdot_qdoti = jd_qd[[0, 2], :]  # only x\z direction
        elif nf == 3:
            jdot_qdoti = jd_qd
        jdot_qdot.append(jdot_qdoti)

    jdot_qdot = jnp.concatenate(jdot_qdot, axis=0)
    return jdot_qdot


# @partial(jit, static_argnums=(4, 5, 6, 7, 8, 9, 10, 11))
def calc_contact_jdot_qdot_core(
    x_tree, q, qdot, contactpoint, idcontact, flag_contact,
    parent, jtype, jaxis, nb, nc, nf):
    jdot_qdot = []
    qddot = jnp.zeros((nb,))
    for i in range(nc):
        jdot_qdoti = jnp.empty((0, 1))
        if flag_contact[i] != 0.0:
            jd_qd = calc_point_acceleration_core(
                x_tree, parent, jtype, jaxis, idcontact[i], q, qdot, qddot, contactpoint[i])

            if nf == 2:
                jdot_qdoti = jd_qd[[0, 2], :]  # only x\z direction
            elif nf == 3:
                jdot_qdoti = jd_qd

        jdot_qdot.append(jdot_qdoti)
    jdot_qdot = jnp.concatenate(jdot_qdot, axis=0)
    return jdot_qdot


def calc_contact_jdot_qdot(model: dict, q: np.ndarray, qdot: np.ndarray, flag_contact: np.ndarray) -> np.ndarray:
    nc = int(model["nc"])
    nb = int(model["nb"])
    nf = int(model["nf"])
    x_tree = model["x_tree"]
    contactpoint = model["contactpoint"]
    idcontact = tuple(model["idcontact"])
    parent = tuple(model["parent"])
    jtype = tuple(model["jtype"])
    jaxis = xyz2int(model["jaxis"])
    contactpoint = model["contactpoint"]
    flag_contact = flag_contact
    jdot_qdot = calc_contact_jdot_qdot_core(
        x_tree, q, qdot, contactpoint, idcontact, flag_contact,
        parent, jtype, jaxis, nb, nc, nf)

    return jdot_qdot

                
