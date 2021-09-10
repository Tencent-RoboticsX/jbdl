from functools import partial
from jbdl.rbdl.math import x_rotx, x_roty, x_rotz, x_trans
import jax.numpy as jnp
from jax.api import jit


@partial(jit, static_argnums=(0, 1))
def joint_model(jtype: int, jaxis: int, q: float):

    if jtype == 0:
        # revolute joint
        # print("revolute joint")
        if jaxis == 0:
            xj = x_rotx(q)
            s = jnp.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        if jaxis == 1:
            xj = x_roty(q)
            s = jnp.array([[0.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
        if jaxis == 2:
            xj = x_rotz(q)
            s = jnp.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])
        if jaxis == 3:
            xj = x_rotx(-q)
            s = jnp.array([[-1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        if jaxis == 4:
            xj = x_roty(-q)
            s = jnp.array([[0.0], [-1.0], [0.0], [0.0], [0.0], [0.0]])
        if jaxis == 5:
            xj = x_rotz(-q)
            s = jnp.array([[0.0], [0.0], [-1.0], [0.0], [0.0], [0.0]])

    if jtype == 1:
        # prismatic joint
        # print("prismatic joint")
        # print(jaxis)
        if jaxis == 0:
            xj = x_trans(jnp.array([[q], [0.0], [0.0]]))
            s = jnp.array([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]])
        if jaxis == 1:
            xj = x_trans(jnp.array([[0.0], [q], [0.0]]))
            s = jnp.array([[0.0], [0.0], [0.0], [0.0], [1.0], [0.0]])
        if jaxis == 2:
            xj = x_trans(jnp.array([[0.0], [0.0], [q]]))
            s = jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])
        if jaxis == 3:
            xj = x_trans(jnp.array([[-q], [0.0], [0.0]]))
            s = jnp.array([[0.0], [0.0], [0.0], [-1.0], [0.0], [0.0]])
        if jaxis == 4:
            xj = x_trans(jnp.array([[0.0], [-q], [0.0]]))
            s = jnp.array([[0.0], [0.0], [0.0], [0.0], [-1.0], [0.0]])
        if jaxis == 5:
            xj =x_trans(jnp.array([[0.0], [0.0], [-q]]))
            s = jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0], [-1.0]])

    return (xj, s)

if __name__ == "__main__":
    from jax import make_jaxpr
    import math
    print(make_jaxpr(joint_model, static_argnums=(0, 1))(0, 'y', math.pi))
    print(joint_model(0, 'y', math.pi))
