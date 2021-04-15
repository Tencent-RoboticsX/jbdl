import jax.numpy as jnp
from jax.api import jit

@jit
def Xrotz(theta):  
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    X = jnp.array([[c, s, 0.0, 0.0, 0.0, 0.0],
                  [-s, c, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, c, s, 0.0],
                  [0.0, 0.0, 0.0, -s, c, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    return X

if __name__ == "__main__":
    import math
    from jax import make_jaxpr
    print(make_jaxpr(Xrotz)(math.pi))
    print(Xrotz(math.pi))