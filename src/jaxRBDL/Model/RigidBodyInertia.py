from jax.api import jit
import jax.numpy as jnp

@jit
def RigidBodyInertia(m: float, c, I):

    flatten_c = jnp.reshape(c, (-1,))
    C = jnp.array(
        [[0.0, -flatten_c[2], flatten_c[1]],
         [flatten_c[2], 0.0, -flatten_c[0]],
         [-flatten_c[1], flatten_c[0], 0.0]])
    rbi = jnp.vstack(
        [jnp.hstack([I + m * jnp.matmul(C, jnp.transpose(C)), m * C]),
         jnp.hstack([m * jnp.transpose(C), m * jnp.eye(3)])])

    return rbi

if __name__ == "__main__":
    from jax import random
    from jax import make_jaxpr
    key = random.PRNGKey(0)
    m = 1.0
    c = jnp.array([1.0, 0.0, 0.0])
    I = random.normal(key, (3, 3))
    print(make_jaxpr(RigidBodyInertia)(m, c, I))
    print(RigidBodyInertia(m, c, I))