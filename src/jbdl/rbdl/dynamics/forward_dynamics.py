import numpy as np
import jax.numpy as jnp
from jax.api import jit
from jbdl.rbdl.model import joint_model
from jbdl.rbdl.math import cross_motion_space, cross_force_space
from functools import partial
from jbdl.rbdl.utils import xyz2int


@partial(jit, static_argnums=(2, 3, 4, 5))
def forward_dynamics_core(x_tree, I, parent, jtype, jaxis, nb, q, qdot, tau, a_grav):  
    S = []
    Xup = []
    v = []
    c = []
    pA = []
    IA = I.copy()

    for i in range(nb):
        XJ, Si = joint_model(jtype[i], jaxis[i], q[i])
        S.append(Si)
        vJ = jnp.multiply(S[i], qdot[i])
        Xup.append(jnp.matmul(XJ, x_tree[i]))
        if parent[i] == 0:
            v.append(vJ)
            c.append(jnp.zeros((6, 1)))
        else:
            v.append(jnp.add(jnp.matmul(Xup[i], v[parent[i]-1]), vJ))
            c.append(jnp.matmul(cross_motion_space(v[i]), vJ))
        pA.append(jnp.matmul(cross_force_space(v[i]), jnp.matmul(IA[i], v[i])))



    U = [jnp.empty((0,))] * nb
    d = [jnp.empty((0,))] * nb
    u = [jnp.empty((0,))] * nb 

    for i in range(nb-1, -1, -1):
        U[i] = jnp.matmul(IA[i], S[i])
        d[i] = jnp.squeeze(jnp.matmul(S[i].transpose(), U[i]))
        u[i] = tau[i] - jnp.squeeze(jnp.matmul(S[i].transpose(), pA[i]))
        if parent[i] != 0:
            Ia = IA[i] - jnp.matmul(U[i]/ d[i], jnp.transpose(U[i]))
            pa = pA[i] + jnp.matmul(Ia, c[i]) + jnp.multiply(U[i], u[i]) / d[i]
            IA[parent[i] - 1] = IA[parent[i] - 1] + jnp.matmul(jnp.matmul(jnp.transpose(Xup[i]), Ia), Xup[i])
            pA[parent[i] - 1] = pA[parent[i] - 1] + jnp.matmul(jnp.transpose(Xup[i]), pa)


    a = []
    qddot = []

    for i in range(nb):
        if parent[i] == 0:
            a.append(jnp.matmul(Xup[i],-a_grav) + c[i])
        else:
            a.append(jnp.matmul(Xup[i], a[parent[i] - 1]) + c[i])
        qddot.append((u[i] - jnp.squeeze(jnp.matmul(jnp.transpose(U[i]), a[i])))/d[i])
        
        a[i] = a[i] + jnp.multiply(S[i],  qddot[i])

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
    I = model["I"]

    qddot = forward_dynamics_core(x_tree, I, tuple(parent), tuple(jtype), jaxis, nb, q, qdot, tau, a_grav)
    return qddot





            

