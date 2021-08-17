import jax.numpy as jnp
from jax.api import jit


@jit
def transform_to_position(x):  
    e = x[0:3, 0:3]
    rx = -jnp.matmul(jnp.transpose(e), x[3:6, 0:3])
    r = jnp.reshape(jnp.array([-rx[1, 2], rx[0, 2], -rx[0, 1]]), (3, 1))
    return r


if __name__ == "__main__":
    from jax import random
    from jax import make_jaxpr
    key = random.PRNGKey(0)
    mat = random.normal(key, (6, 6))
    print(make_jaxpr(transform_to_position)(mat))
    print(transform_to_position(mat))