from jax import core, dtypes, lax
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax.interpreters import xla, ad, batching
from jax.abstract_arrays import ShapedArray
from jbdl.experimental.custom_ops.trace import trace, expectNotImplementedError
from jax.lib import xla_client
from jbdl.experimental import cpu_ops
from jax.api import jit
from jax.api import jacfwd, device_put


# Register the CPU XLA custom calls

for name, value in cpu_ops.registrations().items():
    if "lcp" in name:
        xla_client.register_cpu_custom_call_target(name, value)

lcp_prim = core.Primitive("lcp")
lcp_prim.multiple_results = True
lcp_prim.def_impl(partial(xla.apply_primitive, lcp_prim))


# This function exposes the primitive to user code and this is the only
# public-facing function in this module
# @trace("lcp")
def lcp(H, f, L, k, lb, ub):
    # We're going to apply array broadcasting here since the logic of our op
    # is much simpler if we require the inputs to all have the same shapes

    x, _ = lcp_prim.bind(H, f, L, k, lb, ub)
    return x


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
# @trace("lcp_abstract_eval")
def lcp_abstract_eval(H, f, L, k, lb, ub):
    nV = f.shape[0]
    nC = k.shape[0]
    dtype = dtypes.canonicalize_dtype(H.dtype)
    assert dtypes.canonicalize_dtype(f.dtype) == dtype
    assert dtypes.canonicalize_dtype(L.dtype) == dtype
    assert dtypes.canonicalize_dtype(k.dtype) == dtype
    assert dtypes.canonicalize_dtype(lb.dtype) == dtype
    assert dtypes.canonicalize_dtype(ub.dtype) == dtype
    assert H.shape == (nV, nV)
    assert f.shape == (nV, 1)
    assert L.shape == (nC, nV)
    assert k.shape == (nC, 1)
    assert lb.shape == (nV, 1)
    assert ub.shape == (nV, 1)
    primal_shape = (nV, 1)
    dual_shape = (nC + 2 * nV, 1)
    return (ShapedArray(primal_shape, dtype), ShapedArray(dual_shape, dtype))


# We also need a translation rule to convert the function into an XLA op. In
# our case this is the custom XLA op that we've written. We're wrapping two
# translation rules into one here: one for the CPU and one for the GPU
# @trace("lcp_translation")
def cpu_lcp_translation(c, H, f, L, k, lb, ub):
    # The inputs have "shapes" that provide both the shape and the dtype
    H_shape = c.get_shape(H)
    f_shape = c.get_shape(f)
    L_shape = c.get_shape(L)
    k_shape = c.get_shape(k)
    lb_shape = c.get_shape(lb)
    ub_shape = c.get_shape(ub)

    # Extract the dtype and shape
    dtype = H_shape.element_type()
    nV, _ = f_shape.dimensions()
    nC, _ = k_shape.dimensions()
    assert f_shape.element_type() == dtype
    assert L_shape.element_type() == dtype
    assert k_shape.element_type() == dtype
    assert lb_shape.element_type() == dtype
    assert ub_shape.element_type() == dtype
    assert H_shape.dimensions() == (nV, nV)
    assert f_shape.dimensions() == (nV, 1)
    assert L_shape.dimensions() == (nC, nV)
    assert k_shape.dimensions() == (nC, 1)
    assert lb_shape.dimensions() == (nV, 1)
    assert ub_shape.dimensions() == (nV, 1)




    # The input specification
    nV_array_shape = xla_client.Shape.array_shape(
        np.dtype(np.int64), (), ())
    nC_array_shape = xla_client.Shape.array_shape(
        np.dtype(np.int64), (), ())
    H_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nV, nV), (1, 0))
    f_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nV, 1), (1, 0))
    L_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nC, nV), (1, 0))
    k_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nC, 1), (1, 0))
    lb_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nV, 1), (1, 0))
    ub_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nV, 1), (1, 0))

    # The output specification
    primal_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nV, 1), (1, 0))
    dual_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nC + 2 * nV, 1), (1, 0))

    



    # We dispatch a different call depending on the dtype
    if dtype == np.float32:
        op_name = b"cpu_lcp_f32"
    elif dtype == np.float64:
        op_name = b"cpu_lcp_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    return xla_client.ops.CustomCallWithLayout(
        c,
        op_name,
        # The inputs:
        operands=(xla_client.ops.ConstantLiteral(c, nV), xla_client.ops.ConstantLiteral(c, nC), H, f, L, k, lb, ub),
        # The input shapes:
        operand_shapes_with_layout=(
            nV_array_shape,
            nC_array_shape,
            H_array_shape,
            f_array_shape,
            L_array_shape,
            k_array_shape,
            lb_array_shape,
            ub_array_shape
        ),
        # The output shapes:
        shape_with_layout=xla_client.Shape.tuple_shape((primal_array_shape, dual_array_shape)),
    )



lcp_prim.def_abstract_eval(lcp_abstract_eval)
# Connect the XLA translation rules for JIT compilation
xla.backend_specific_translations["cpu"][lcp_prim] = cpu_lcp_translation

def lcp_kkt(x, z,  H, f, L, k, lb, ub):
    n_var = H.shape[1]

    L = jnp.vstack([L, -np.eye(n_var)])
    k = jnp.vstack([k, -lb])
    L = jnp.vstack([L, np.eye(n_var)])
    k = jnp.vstack([k, ub])

    lagrange = jnp.matmul(H, x) + f + jnp.matmul(jnp.transpose(L),  z)
    inequality = (jnp.matmul(L, x) - k) * z
    kkt = jnp.vstack([lagrange, inequality])
    return kkt

# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************

# In this case we don't need to define a transpose rule in order to support
# reverse and higher order differentiation. This might not be true in other
# applications, so check out the "How JAX primitives work" tutorial in the JAX
# documentation for more info as necessary.
def lcp_jvp(arg_values, arg_tangents):
    H, f, L, k, lb, ub = arg_values
    H_dot, f_dot, L_dot, k_dot, lb_dot, ub_dot = arg_tangents

    # We use "bind" here because we don't want to mod the mean anomaly again
    x_star, z_star = lcp_prim.bind(H, f, L, k, lb, ub)
    nV = H.shape[1]


    dkkt2dx = jacfwd(lcp_kkt, argnums=0)(x_star, z_star,  H, f, L, k, lb, ub)
    dkkt2dz = jacfwd(lcp_kkt, argnums=1)(x_star, z_star,  H, f, L, k, lb, ub)
   
    dkkt2dH = jacfwd(lcp_kkt, argnums=2)(x_star, z_star,  H, f, L, k, lb, ub)
    dkkt2df = jacfwd(lcp_kkt, argnums=3)(x_star, z_star,  H, f, L, k, lb, ub)
    dkkt2dL = jacfwd(lcp_kkt, argnums=4)(x_star, z_star,  H, f, L, k, lb, ub)
    dkkt2dk = jacfwd(lcp_kkt, argnums=5)(x_star, z_star,  H, f, L, k, lb, ub)
    dkkt2dlb = jacfwd(lcp_kkt, argnums=6)(x_star, z_star,  H, f, L, k, lb, ub)
    dkkt2dub = jacfwd(lcp_kkt, argnums=7)(x_star, z_star,  H, f, L, k, lb, ub)
   
    dkkt2dxz = jnp.concatenate([dkkt2dx, dkkt2dz], axis=2)
    dkkt2dxz = jnp.transpose(dkkt2dxz, [3, 1, 0, 2])
    dkkt2dH = jnp.transpose(dkkt2dH, [2, 3, 0, 1])
    dkkt2df = jnp.transpose(dkkt2df, [2, 3, 0, 1])
    dkkt2dL = jnp.transpose(dkkt2dL, [2, 3, 0, 1])
    dkkt2dk = jnp.transpose(dkkt2dk, [2, 3, 0, 1])
    dkkt2dlb = jnp.transpose(dkkt2dlb, [2, 3, 0, 1])
    dkkt2dub = jnp.transpose(dkkt2dub, [2, 3, 0, 1])

    dxz2dH = -jnp.linalg.solve(dkkt2dxz, dkkt2dH)
    dxz2dH = jnp.transpose(dxz2dH, [2, 3, 0, 1])
    # dx2dH = dxz2dH[0:nV, ...]
    # dz2dH = dxz2dH[nV:, ...]
    dxz2df =  -jnp.linalg.solve(dkkt2dxz, dkkt2df)
    dxz2df = jnp.transpose(dxz2df, [2, 3, 0, 1])
    # dx2df = dxz2df[0:nV, ...]
    # dz2df = dxz2df[nV:, ...]
    dxz2dL = -jnp.linalg.solve(dkkt2dxz, dkkt2dL)
    dxz2dL = jnp.transpose(dxz2dL, [2, 3, 0, 1])
    # dx2dL = dxz2dL[0:nV, ...]
    # dz2dL = dxz2dL[nV:, ...]
    dxz2dk = -jnp.linalg.solve(dkkt2dxz, dkkt2dk)
    dxz2dk = jnp.transpose(dxz2dk, [2, 3, 0, 1])
    # dx2dk = dxz2dk[0:nV, ...]
    # dz2dk = dxz2dk[nV:, ...]
    dxz2dlb = -jnp.linalg.solve(dkkt2dxz, dkkt2dlb)
    dxz2dlb = jnp.transpose(dxz2dlb, [2, 3, 0, 1])
    # dx2dlb = dxz2dlb[0:nV, ...]
    # dz2dlb = dxz2df[nV:, ...]
    dxz2dub = -jnp.linalg.solve(dkkt2dxz, dkkt2dub)
    dxz2dub = jnp.transpose(dxz2dub, [2, 3, 0, 1])
    # dx2dub = dxz2dub[0:nV, ...]
    # dz2dub = dxz2dub[nV:, ...]


    if type(H_dot) is  ad.Zero:
        diff_H = device_put(0.0)
    else:
        diff_H = jnp.sum(dxz2dH * H_dot, axis=(-2, -1))

    if type(f_dot) is ad.Zero:
        diff_f = device_put(0.0)
    else:
        diff_f = jnp.sum(dxz2df * f_dot, axis=(-2, -1))

    if type(L_dot) is ad.Zero:
        diff_L = device_put(0.0)
    else:
        diff_L = jnp.sum(dxz2dL * L_dot, axis=(-2, -1))

    if type(k_dot) is ad.Zero:
        diff_k = device_put(0.0)
    else:
        diff_k = jnp.sum(dxz2dk * k_dot, axis=(-2, -1))
    
    if type(lb_dot) is ad.Zero:
        diff_lb = device_put(0.0)
    else:
        diff_lb = jnp.sum(dxz2dlb * lb_dot, axis=(-2, -1))

    if type(ub_dot) is ad.Zero:
        diff_ub = device_put(0.0)
    else:
        diff_ub = jnp.sum(dxz2dub * ub_dot, axis=(-2, -1))
    
    diff = diff_H + diff_f + diff_L + diff_k + diff_lb + diff_ub

    return (x_star, z_star), (diff[0:nV, ...], diff[nV:,...])

ad.primitive_jvps[lcp_prim] = lcp_jvp

  



if __name__ == "__main__":
    from jax.config import config
    config.update("jax_enable_x64", True)
    H = np.array([1.0, -1.0, -1.0, 2.0]).reshape(2, 2)
    f = np.array([-2.0, -6.0]).reshape(2, 1)
    L = np.array([1.0, 1.0, -1.0, 2.0, 2.0, 1.0]).reshape(3, 2)
    k = np.array([2.0, 2.0, 3.0]).reshape(3, 1)
    lb = np.array([0.0, 0.0]).reshape(2, 1)
    ub = np.array([0.5, 5.0 ]).reshape(2, 1)
    # print(H.dtype)
    # with expectNotImplementedError():
    print(jit(lcp)(H, f, L, k, lb, ub))

    fwd = jacfwd(lcp, argnums=2)(H, f, L, k, lb, ub)
    print(fwd)
    print(fwd.shape)

    from jax import random

    seed = 1701
    num_steps = 100
    key = random.PRNGKey(seed)
    sum = 0.0
    for i in range(num_steps):
        key, subkey = random.split(key)
        x = random.randint(subkey, (4, ), 0, 2)
        print(x)
        sum += lax.cond(
            jnp.sum(x),
            lambda _: jnp.ones((2, 1)),
            lambda _: lcp(H, f, L, k, lb, ub),
            operand = None
        )
        print(sum)

    print(sum)
        
       
    # print(fwd0)
    # print(fwd0.shape)
    # print(fwd1)
    # print(fwd1.shape)

    
