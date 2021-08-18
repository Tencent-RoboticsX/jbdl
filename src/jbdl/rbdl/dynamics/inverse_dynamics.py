from functools import partial
import jax.numpy as jnp
from jax.api import jit
from jbdl.rbdl.model import joint_model
from jbdl.rbdl.math import cross_motion_space, cross_force_space
from jbdl.rbdl.utils import xyz2int


@partial(jit, static_argnums=(2, 3, 4, 5))
def inverse_dynamics_core(x_tree, inertia, parent, jtype, jaxis, nb, q, qdot, qddot, a_grav):
    s = []
    x_up = []
    v = []
    avp = []
    fvp = []

    for i in range(nb):
        xj, si = joint_model(jtype[i], jaxis[i], q[i])
        s.append(si)
        vj = jnp.multiply(s[i], qdot[i])
        x_up.append(jnp.matmul(xj, x_tree[i]))
        if parent[i] == 0:
            v.append(vj)
            avp.append(jnp.matmul(x_up[i], -a_grav) + s[i] * qddot[i])
        else:
            v.append(jnp.matmul(x_up[i], v[parent[i] - 1]) + vj)
            avp.append(
                jnp.matmul(x_up[i], avp[parent[i] - 1]) \
                    + jnp.multiply(s[i], qddot[i]) + jnp.matmul(cross_motion_space(v[i]), vj))
        fvp.append(
            jnp.matmul(inertia[i], avp[i]) \
                + jnp.matmul(jnp.matmul(cross_force_space(v[i]), inertia[i]), v[i]))

    tau = [0.0] * nb

    for i in range(nb-1, -1, -1):
        tau[i] = jnp.squeeze(jnp.matmul(jnp.transpose(s[i]), fvp[i]))
        if parent[i] != 0:
            fvp[parent[i] - 1] = fvp[parent[i] - 1] + jnp.matmul(jnp.transpose(x_up[i]), fvp[i])
    tau = jnp.reshape(jnp.array(tau), (nb, 1))

    return tau


def inverse_dynamics(model, q, qdot, qddot):

    a_grav = model["a_grav"]
    qdot = qdot.flatten()
    qddot = qddot.flatten()
    nb = model["nb"]
    jtype = tuple(model["jtype"])
    jaxis = xyz2int(model["jaxis"])
    parent = tuple(model["parent"])
    x_tree = model["x_tree"]
    inertia = model["inertia"]

    tau = inverse_dynamics_core(x_tree, inertia, tuple(parent), tuple(jtype), jaxis, nb, q, qdot, qddot, a_grav)
    return tau
