from jax.api import jit
import jax.numpy as jnp


@jit
def cross_matrix(v):
    flatten_v = jnp.reshape(v, (-1,))
    cross_mat = jnp.array(
        [[0.0, -flatten_v[2], flatten_v[1]],
         [flatten_v[2], 0.0, -flatten_v[0]],
         [-flatten_v[1], flatten_v[0], 0.0]])
    return cross_mat


if __name__ == "__main__":
    from jax import make_jaxpr
    v = jnp.ones((3, 1))
    print(make_jaxpr(cross_matrix)(v))
    print(cross_matrix(v))
