import jax.numpy as jnp
from jax.api import jit


@jit
def inverse_motion_space(x): 
    e = x[0:3, 0:3]
    r = x[3:6, 0:3]
    x_inv = jnp.vstack(
        [jnp.hstack([jnp.transpose(e), jnp.zeros((3, 3))]),
         jnp.hstack([jnp.transpose(r), jnp.transpose(e)])])
    return x_inv


if __name__ == "__main__":
    from jax import random
    from jax import make_jaxpr
    key = random.PRNGKey(0)
    x = random.normal(key, (6, 6))
    print(make_jaxpr(inverse_motion_space)(x))
    print(inverse_motion_space(x))
