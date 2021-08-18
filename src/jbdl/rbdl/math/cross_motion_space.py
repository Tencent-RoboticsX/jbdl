from jax.api import jit
import jax.numpy as jnp


@jit
def cross_motion_space(v):
    flatten_v = jnp.reshape(v, (-1,))
    vcross = jnp.array([[0.0, -flatten_v[2], flatten_v[1], 0.0, 0.0, 0.0],
                       [flatten_v[2], 0.0, -flatten_v[0], 0.0, 0.0, 0.0],
                       [-flatten_v[1], flatten_v[0], 0.0, 0.0, 0.0, 0.0],
                       [0.0, -flatten_v[5], flatten_v[4], 0.0, -flatten_v[2], flatten_v[1]],
                       [flatten_v[5], 0.0, -flatten_v[3], flatten_v[2], 0.0, -flatten_v[0]],
                       [-flatten_v[4], flatten_v[3], 0.0, -flatten_v[1], flatten_v[0], 0.0]])
    return vcross


if __name__ == "__main__":
    from jax import make_jaxpr
    v = jnp.ones((6,1))
    print(make_jaxpr(cross_motion_space)(v))
    print(cross_motion_space(v))
