import jax.numpy as jnp
from jax.api import jit
from jaxBDL.rbdl.math import cross_motion_space

@jit
def cross_force_space(v):
    vcross = -jnp.transpose(cross_motion_space(v))
    return vcross


if __name__ == "__main__":
    from jax import make_jaxpr
    a = jnp.ones((6, 1))
    print(make_jaxpr(cross_force_space)(a))
    print(cross_force_space(a))

