from jax.api import jit
import jax.numpy as jnp

@jit
def x_trans(r):
    flatten_r = jnp.reshape(r, (-1,))
    X = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
         [0.0, flatten_r[2], -flatten_r[1], 1.0, 0.0, 0.0],
         [-flatten_r[2], 0.0, flatten_r[0], 0.0,  1.0,  0.0],
         [flatten_r[1], -flatten_r[0], 0.0, 0.0, 0.0, 1.0]])
    return X

if __name__ == "__main__":
    from jax import make_jaxpr
    v = jnp.array([[1.0], [2.0], [3.0]])
    print(make_jaxpr(x_trans)(v))
    print(x_trans(v))

