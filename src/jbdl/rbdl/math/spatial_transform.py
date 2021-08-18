from jax.api import jit
import jax.numpy as jnp
from jbdl.rbdl.math import cross_matrix


@jit
def spatial_transform(e, r):
    x_transform = jnp.vstack([jnp.hstack([e, jnp.zeros((3, 3))]), jnp.hstack([jnp.matmul(-e, cross_matrix(r)), e])])
    return x_transform


if __name__ == "__main__":
    from jax import random
    from jax import make_jaxpr
    key = random.PRNGKey(0)
    e = random.normal(key, (3, 3))
    r = random.normal(key, (3,))
    print(make_jaxpr(spatial_transform)(e, r))
    print(spatial_transform(e, r))
    