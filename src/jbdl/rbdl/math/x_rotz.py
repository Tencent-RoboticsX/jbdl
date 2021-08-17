import jax.numpy as jnp
from jax.api import jit

@jit
def x_rotz(theta):  
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    x = jnp.array([[c, s, 0.0, 0.0, 0.0, 0.0],
                  [-s, c, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, c, s, 0.0],
                  [0.0, 0.0, 0.0, -s, c, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    return x

if __name__ == "__main__":
    import math
    from jax import make_jaxpr
    print(make_jaxpr(x_rotz)(math.pi))
    print(x_rotz(math.pi))
