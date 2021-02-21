import jax.numpy as jnp
from jax.api import jit
from jaxRBDL.Math.CrossMotionSpace import CrossMotionSpace

@jit
def CrossForceSpace(v):
    vcross = -jnp.transpose(CrossMotionSpace(v))
    return vcross


if __name__ == "__main__":
    from jax import make_jaxpr
    a = jnp.ones((6, 1))
    print(make_jaxpr(CrossForceSpace)(a))
    print(CrossForceSpace(a))

