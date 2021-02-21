from jaxRBDL.Math.Xrotx import Xrotx
from jaxRBDL.Math.Xroty import Xroty
from jaxRBDL.Math.Xrotz import Xrotz
from jaxRBDL.Math.Xtrans import Xtrans
import jax.numpy as jnp
from functools import partial
from jax.api import jit


@partial(jit, static_argnums=(0, 1))
def JointModel(jtype: int, jaxis: str, q: float):
    if jtype == 0:
        # revolute joint
        if jaxis == 'x':
            Xj = Xrotx(q)
            S = jnp.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        if jaxis == 'y':
            Xj = Xroty(q)
            S = jnp.array([[0.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
        if jaxis == 'z':
            Xj = Xrotz(q)
            S = jnp.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])
    if jtype == 1:
        # prismatic joint
        if jaxis == 'x':
            Xj = Xtrans(jnp.array([[q], [0.0], [0.0]]))
            S = jnp.array([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]])
        if jaxis == 'y':
            Xj = Xtrans(jnp.array([[0.0], [q], [0.0]]))
            S = jnp.array([[0.0], [0.0], [0.0], [0.0], [1.0], [0.0]])
        if jaxis == 'z':
            Xj = Xtrans(jnp.array([[0.0], [0.0], [q]]))
            S = jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])
    
    return (Xj, S)

if __name__ == "__main__":
    from jax import make_jaxpr
    import math
    import numpy as np
    print(make_jaxpr(JointModel, static_argnums=(0, 1))(0, 'y', math.pi))
    print(JointModel(0, 'y', math.pi))