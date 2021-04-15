from ast import iter_child_nodes
from typing import List
from jax._src.lax.lax import sub
import numpy as np
import jax.numpy as jnp
from jaxBDL.rbdl.model import joint_model

from functools import partial
from jax.api import jit
from jax import lax


@partial(jit, static_argnums=(2, 3, 4, 5))
def composite_rigid_body_algorithm_core(Xtree, I, parent, jtype, jaxis, NB, q):
    # print("Re-Tracing")

    IC = I.copy()
    
    S = []
    Xup = []

    for i in range(NB):
        XJ, Si = joint_model(jtype[i], jaxis[i], q[i])
        S.append(Si)
        Xup.append(jnp.matmul(XJ, Xtree[i]))


    for j in range(NB-1, -1, -1):
        if parent[j] != 0:
            IC[parent[j] - 1] = IC[parent[j] - 1] + jnp.matmul(jnp.matmul(Xup[j].transpose(), IC[j]), Xup[j])


    H = jnp.zeros((NB, NB))

    for i in range(NB):
        fh = jnp.matmul(IC[i],  S[i])
        H = H.at[i, i].set(jnp.squeeze(jnp.matmul(S[i].transpose(), fh)))
        j = i
        while parent[j] > 0:
            fh = jnp.matmul(Xup[j].transpose(), fh)
            j = parent[j] - 1
            H = H.at[i,j].set(jnp.squeeze(jnp.matmul(S[j].transpose(), fh)))
            H = H.at[j,i].set(H[i,j])

    return H




def composite_rigid_body_algorithm(model: dict, q):

    NB = int(model["NB"])
    jtype = model["jtype"]
    jaxis = model['jaxis']
    parent = model['parent']
    Xtree = model["Xtree"]
    I = model["I"]

    q = q.flatten()
    H = composite_rigid_body_algorithm_core(Xtree, I, tuple(parent), tuple(jtype), jaxis, NB, q)

    return H






