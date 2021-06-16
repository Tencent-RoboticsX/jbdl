from jax.api import jit
import jax.numpy as jnp

@jit
def init_Ic_by_cholesky(l):
    """
    Args:
        l (jnp.Array): float(6,) non-zero entris of a upper triangle matrix: exp(lxx) lxy lxz exp(lyy) lyz exp(lzz)

    """
    flatten_l = jnp.reshape(l, (-1,))
    L = jnp.array(
        [[jnp.exp(flatten_l[0]), flatten_l[1], flatten_l[2]],
         [0.0, jnp.exp(flatten_l[3]), flatten_l[4]],
         [0.0, 0.0, jnp.exp(flatten_l[5])]])

    Ic = jnp.matmul(jnp.transpose(L),  L)
    return Ic

@jit
def rigid_body_inertia(m: float, c, I):

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
    m = 2.0
    c = jnp.array([0.0, 0.0, 0.0])
    I = init_Ic_by_cholesky(jnp.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]))
    print(make_jaxpr(rigid_body_inertia)(m, c, I))
    print(rigid_body_inertia(m, c, I))