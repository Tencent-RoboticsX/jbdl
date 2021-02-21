from jax.api import jit
import jax.numpy as jnp
from jaxRBDL.Math.CrossMatrix import CrossMatrix

@jit
def SpatialTransform(E, r):   
    col_r = jnp.reshape(r, (3, 1))
    X_T = jnp.vstack([jnp.hstack([E, jnp.zeros((3, 3))]), jnp.hstack([jnp.matmul(-E, CrossMatrix(r)), E])])
    return X_T


if __name__ == "__main__":
    from jax import random
    from jax import make_jaxpr
    key = random.PRNGKey(0)
    E = random.normal(key, (3, 3))
    r = random.normal(key, (3,))
    print(make_jaxpr(SpatialTransform)(E, r))
    print(SpatialTransform(E, r))
    