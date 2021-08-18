from functools import partial
import jax.numpy as jnp
from jax.api import jit
from jbdl.rbdl.model import joint_model
from jbdl.rbdl.math import cross_motion_space, cross_force_space
from jbdl.rbdl.utils import xyz2int


@partial(jit, static_argnums=(2, 3, 4, 5))
def forward_dynamics_core(x_tree, inertia, parent, jtype, jaxis, nb, q, qdot, tau, a_grav):
    s = []
    x_up = []
    v = []
    c = []
    p_aa = []
    inertia_aa = inertia.copy()

    for i in range(nb):
        xj, si = joint_model(jtype[i], jaxis[i], q[i])
        s.append(si)
        vj = jnp.multiply(s[i], qdot[i])
        x_up.append(jnp.matmul(xj, x_tree[i]))
        if parent[i] == 0:
            v.append(vj)
            c.append(jnp.zeros((6, 1)))
        else:
            v.append(jnp.add(jnp.matmul(x_up[i], v[parent[i]-1]), vj))
            c.append(jnp.matmul(cross_motion_space(v[i]), vj))
        p_aa.append(jnp.matmul(cross_force_space(v[i]), jnp.matmul(inertia_aa[i], v[i])))



    uu = [jnp.empty((0,))] * nb
    d = [jnp.empty((0,))] * nb
    u = [jnp.empty((0,))] * nb

    for i in range(nb-1, -1, -1):
        uu[i] = jnp.matmul(inertia_aa[i], s[i])
        d[i] = jnp.squeeze(jnp.matmul(s[i].transpose(), uu[i]))
        u[i] = tau[i] - jnp.squeeze(jnp.matmul(s[i].transpose(), p_aa[i]))
        if parent[i] != 0:
            ia = inertia_aa[i] - jnp.matmul(uu[i] / d[i], jnp.transpose(uu[i]))
            pa = p_aa[i] + jnp.matmul(ia, c[i]) + jnp.multiply(uu[i], u[i]) / d[i]
            inertia_aa[parent[i] - 1] = inertia_aa[parent[i] - 1] \
                + jnp.matmul(jnp.matmul(jnp.transpose(x_up[i]), ia), x_up[i])
            p_aa[parent[i] - 1] = p_aa[parent[i] - 1] + jnp.matmul(jnp.transpose(x_up[i]), pa)

    a = []
    qddot = []

    for i in range(nb):
        if parent[i] == 0:
            a.append(jnp.matmul(x_up[i], -a_grav) + c[i])
        else:
            a.append(jnp.matmul(x_up[i], a[parent[i] - 1]) + c[i])

        qddot.append((u[i] - jnp.squeeze(jnp.matmul(jnp.transpose(uu[i]), a[i])))/d[i])
        a[i] = a[i] + jnp.multiply(s[i], qddot[i])

    qddot = jnp.reshape(jnp.stack(qddot), (nb, ))
    return qddot


def forward_dynamics(model, q, qdot, tau):
    q = q.flatten()
    qdot = qdot.flatten()
    tau = tau.flatten()
    a_grav = model["a_grav"]
    nb = model["nb"]
    jtype = tuple(model["jtype"])
    jaxis = xyz2int(model["jaxis"])
    parent = tuple(model["parent"])
    x_tree = model["x_tree"]
    inertia = model["inertia"]

    qddot = forward_dynamics_core(x_tree, inertia, tuple(parent), tuple(jtype), jaxis, nb, q, qdot, tau, a_grav)
    return qddot
