from jax.api import jit
import jax.numpy as jnp

@jit
def TransformToPosition(X):  
    E = X[0:3, 0:3]
    rx = -jnp.matmul(jnp.transpose(E), X[3:6, 0:3])
    r = jnp.reshape(jnp.array([-rx[1, 2], rx[0, 2], -rx[0, 1]]), (3, 1))
    return r

if __name__ == "__main__":
    from jax import random
    from jax import make_jaxpr
    key = random.PRNGKey(0)
    mat = random.normal(key, (6, 6))
    print(make_jaxpr(TransformToPosition)(mat))
    print(TransformToPosition(mat))