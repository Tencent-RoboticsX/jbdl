from jax.api import jit
import jax.numpy as jnp


@jit
def Xrotx(theta):
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    X = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, c, s, 0.0, 0.0, 0.0],
         [0.0, -s, c, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, c, s],
         [0.0, 0.0, 0.0, 0.0, -s, c]])
    return X

if __name__ == "__main__":
    import math
    from jax import make_jaxpr
    print(make_jaxpr(Xrotx)(math.pi))
    print(Xrotx(math.pi))