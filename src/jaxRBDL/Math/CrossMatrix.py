from jax.api import jit
import jax.numpy as jnp

@jit
def CrossMatrix(v):
    flatten_v = jnp.reshape(v, (-1,))
    CroMat = jnp.array(
        [[0.0, -flatten_v[2], flatten_v[1]],
         [flatten_v[2], 0.0, -flatten_v[0]],
         [-flatten_v[1], flatten_v[0], 0.0]])
    return CroMat



if __name__ == "__main__":
    from jax import make_jaxpr
    v = jnp.ones((3, 1))
    print(make_jaxpr(CrossMatrix)(v))
    print(CrossMatrix(v))