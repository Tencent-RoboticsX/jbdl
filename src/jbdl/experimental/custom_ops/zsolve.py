from jax import core
import jax.numpy as jnp
import numpy as np


zsolve_prim = core.Primitive("zsolve")
zsolve_prim.multiple_results = True


def zsolve(aa, b):
    return zsolve_prim.bind(aa, b)


def zsolve_impl(aa, b):
    row_sel = np.any(aa, axis=0)
    col_sel = np.any(aa, axis=1)
    assert (row_sel == col_sel).all()
    aa_sel = aa[row_sel, :][:, col_sel]
    b_sel = b[row_sel]
    x_value = np.linalg.solve(aa_sel, b_sel)
    x = np.zeros_like(b)
    x[row_sel, ...] = x_value

    return x


zsolve_prim.def_impl(zsolve_impl)
