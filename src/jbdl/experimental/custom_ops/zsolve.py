from jax import core, dtypes, lax
from jax.interpreters.masking import bind_shapes
from jbdl.experimental.custom_ops.trace import trace, expectNotImplementedError
import jax.numpy as jnp
import numpy as np

zsolve_prim = core.Primitive("zsolve")
zsolve_prim.multiple_results = True

@trace("zsolve")
def zsolve(A, b):
    return zsolve_prim.bind(A, b)

def zsolve_impl(A, b):
    row_sel = np.any(A, axis=0)
    col_sel = np.any(A, axis=1)
    assert (row_sel == col_sel).all()
    A_sel = A[row_sel, :][:, col_sel]
    b_sel = b[row_sel]
    x_value = np.linalg.solve(A_sel, b_sel)
    x = np.zeros_like(b)
    x[row_sel, ...] = x_value

    return x
    
zsolve_prim.def_impl(zsolve_impl)



if __name__ == "__main__":
    from jax.scipy.linalg import block_diag
    import jax
    A = block_diag(jnp.eye(3), jnp.zeros((3, 3)))
    b = jnp.vstack([jnp.ones((3, 1)), jnp.zeros((3, 1))])
    print(A)
    print(b)
    # with expectNotImplementedError(): 
    #     zsolve(A, b)
    def zsolve(A, b):
        row_sel = jnp.any(A, axis=0)
        col_sel = jnp.any(A, axis=1)
        assert (row_sel == col_sel).all()
        A_sel = A[row_sel, :][:, col_sel]
        b_sel = b[row_sel]
        x_value = jnp.linalg.solve(A_sel, b_sel)
        print(x_value)
        x = jnp.zeros_like(b)
        x = jax.ops.index_update(x, jax.ops.index[row_sel, ...], x_value)
        return x

    jax.jit(zsolve)(A, b)




