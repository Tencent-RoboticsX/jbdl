import sys
import numpy as np
import jax
from jax.api import jit
import jax.numpy as jnp
import json
from functools import partial
import time
from jax.config import config

config.update("jax_enable_x64", True)

sys.path.append(".")

from jaxBDL.rbdl.dynamics import inverse_dynamics, inverse_dynamics_core


def I_to_phi(I):
    phi = np.zeros((13 * 10))
    for i in range(13):
        phi = jax.ops.index_update(phi, i * 10, I[i + 5, 3, 3])
        phi = jax.ops.index_update(phi, i * 10 + 1, I[i + 5, 2, 4])
        phi = jax.ops.index_update(phi, i * 10 + 2, I[i + 5, 0, 5])
        phi = jax.ops.index_update(phi, i * 10 + 3, I[i + 5, 1, 3])
        phi = jax.ops.index_update(phi, i * 10 + 4, I[i + 5, 0, 0])
        phi = jax.ops.index_update(phi, i * 10 + 5, I[i + 5, 0, 1])
        phi = jax.ops.index_update(phi, i * 10 + 6, I[i + 5, 0, 2])
        phi = jax.ops.index_update(phi, i * 10 + 7, I[i + 5, 1, 1])
        phi = jax.ops.index_update(phi, i * 10 + 8, I[i + 5, 1, 2])
        phi = jax.ops.index_update(phi, i * 10 + 9, I[i + 5, 2, 2])

    return phi


@jax.jit
def phi_to_I(phi):
    I = np.zeros((13, 6, 6))
    for i in range(13):
        I = jax.ops.index_update(I, ((i, i, i), (3, 4, 5), (3, 4, 5)), phi[i * 10])
        I = jax.ops.index_update(
            I, ((i, i, i, i), (1, 2, 5, 4), (5, 4, 1, 2)), phi[i * 10 + 1]
        )
        I = jax.ops.index_update(
            I, ((i, i, i, i), (0, 2, 5, 3), (5, 3, 0, 2)), phi[i * 10 + 2]
        )
        I = jax.ops.index_update(
            I, ((i, i, i, i), (0, 1, 4, 3), (4, 3, 0, 1)), phi[i * 10 + 3]
        )
        I = jax.ops.index_update(I, (i, 0, 0), phi[i * 10 + 4])
        I = jax.ops.index_update(I, ((i, i), (0, 1), (1, 0)), phi[i * 10 + 5])
        I = jax.ops.index_update(I, ((i, i), (0, 2), (2, 0)), phi[i * 10 + 6])
        I = jax.ops.index_update(I, (i, 1, 1), phi[i * 10 + 7])
        I = jax.ops.index_update(I, ((i, i), (1, 2), (2, 1)), phi[i * 10 + 8])
        I = jax.ops.index_update(I, (i, 2, 2), phi[i * 10 + 9])
        I = jax.ops.index_mul(
            I, ((i, i, i, i, i, i), (0, 4, 2, 3, 1, 5), (4, 0, 3, 2, 5, 1)), -1
        )

    return jnp.concatenate([np.zeros((5, 6, 6)), I], axis=0)


def use_dyn_core(Xtree, phi, parent, jtype, jaxis, NB, q, qdot, qddot, a_grav):
    I = phi_to_I(phi)
    return inverse_dynamics_core(
        Xtree, I, parent, jtype, jaxis, NB, q, qdot, qddot, a_grav
    )


if __name__ == "__main__":
    # The following code is for getting A
    jax_grad = jax.jacfwd(use_dyn_core, argnums=1)
    jax_grad_jit = partial(jit, static_argnums=(2, 3, 4, 5))(jax_grad)
    exp_time = 10000
    # when use, get parameters and then get A
    # get parameters
    with open("new_model.json", "r") as load_f:
        model = json.load(load_f)
    NB = model["NB"]
    q = np.random.rand(NB)
    qdot = np.random.rand(NB)
    qddot = np.random.rand(NB)
    Xtree = np.array(model["Xtree"])
    phi = I_to_phi(np.array(model["I"]))
    parent = tuple(model["parent"])
    jtype = tuple(model["jtype"])
    jaxis = model["jaxis"]
    NB = model["NB"]
    a_grav = np.array(model["a_grav"]).reshape((6, 1))
    # get A
    A_jax = jax_grad_jit(
        Xtree,
        phi,
        parent,
        jtype,
        jaxis,
        NB,
        q,
        qdot,
        qddot,
        a_grav,
    )