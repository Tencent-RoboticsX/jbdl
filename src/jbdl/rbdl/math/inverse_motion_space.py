import jax.numpy as jnp
from jax.api import jit

@jit
def inverse_motion_space(X): 
    E = X[0:3, 0:3]
    r = X[3:6, 0:3]
    Xinv = jnp.vstack([jnp.hstack([jnp.transpose(E), jnp.zeros((3, 3))]),
    jnp.hstack([jnp.transpose(r), jnp.transpose(E)])])
    return Xinv

if __name__ == "__main__":
    from jax import random
    from jax import make_jaxpr
    key = random.PRNGKey(0)
    X = random.normal(key, (6, 6))
    print(make_jaxpr(inverse_motion_space)(X))
    print(inverse_motion_space(X))