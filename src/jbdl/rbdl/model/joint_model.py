from jbdl.rbdl.math import Xrotx, Xroty, Xrotz, Xtrans
import jax.numpy as jnp
from functools import partial
from jax.api import jit


@partial(jit, static_argnums=(0, 1))
def joint_model(jtype: int, jaxis: str, q: float):
    if jtype == 0:
        # revolute joint
        if jaxis == 0:
            Xj = Xrotx(q)
            S = jnp.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        if jaxis == 1:
            Xj = Xroty(q)
            S = jnp.array([[0.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
        if jaxis == 2:
            Xj = Xrotz(q)
            S = jnp.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])
    if jtype == 1:
        # prismatic joint
        if jaxis == 0:
            Xj = Xtrans(jnp.array([[q], [0.0], [0.0]]))
            S = jnp.array([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]])
        if jaxis == 1:
            Xj = Xtrans(jnp.array([[0.0], [q], [0.0]]))
            S = jnp.array([[0.0], [0.0], [0.0], [0.0], [1.0], [0.0]])
        if jaxis == 2:
            Xj = Xtrans(jnp.array([[0.0], [0.0], [q]]))
            S = jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])
    
    return (Xj, S)

if __name__ == "__main__":
    from jax import make_jaxpr
    import math
    import numpy as np
    print(make_jaxpr(joint_model, static_argnums=(0, 1))(0, 1, math.pi))
    print(joint_model(0, 1, math.pi))