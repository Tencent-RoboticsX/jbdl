from functools import partial
import jax.numpy as jnp
from jax.api import jit
from jbdl.rbdl.model import joint_model
from jbdl.rbdl.utils import xyz2int


@partial(jit, static_argnums=(2, 3, 4, 5))
def composite_rigid_body_algorithm_core(x_tree, inertia, parent, jtype, jaxis, nb, q):
    # print("Re-Tracing")

    ic = inertia.copy()
    s = []
    x_up = []

    for i in range(nb):
        xj, si = joint_model(jtype[i], jaxis[i], q[i])
        s.append(si)
        x_up.append(jnp.matmul(xj, x_tree[i]))


    for j in range(nb-1, -1, -1):
        if parent[j] != 0:
            ic[parent[j] - 1] = ic[parent[j] - 1] + \
                jnp.matmul(jnp.matmul(x_up[j].transpose(), ic[j]), x_up[j])

    h = jnp.zeros((nb, nb))

    for i in range(nb):
        fh = jnp.matmul(ic[i],  s[i])
        h = h.at[i, i].set(jnp.squeeze(jnp.matmul(s[i].transpose(), fh)))
        j = i
        while parent[j] > 0:
            fh = jnp.matmul(x_up[j].transpose(), fh)
            j = parent[j] - 1
            h = h.at[i,j].set(jnp.squeeze(jnp.matmul(s[j].transpose(), fh)))
            h = h.at[j,i].set(h[i,j])

    return h


def composite_rigid_body_algorithm(model: dict, q):

    nb = int(model["nb"])
    jtype = tuple(model["jtype"])
    jaxis = xyz2int(model['jaxis'])
    parent = tuple(model['parent'])
    x_tree = model["x_tree"]
    inertia = model["inertia"]

    q = q.flatten()
    h = composite_rigid_body_algorithm_core(
        x_tree, inertia, tuple(parent), tuple(jtype), jaxis, nb, q)

    return h
