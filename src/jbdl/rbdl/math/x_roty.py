import jax.numpy as jnp
from jax.api import jit

@jit
def x_roty(theta):
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    X = jnp.array([[c, 0.0, -s, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [s, 0.0, c, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, c, 0.0, -s],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, s, 0.0, c]])
    return X

if __name__ == "__main__":
    import math
    from jax import make_jaxpr
    print(make_jaxpr(x_roty)(math.pi))
    print(x_roty(math.pi))