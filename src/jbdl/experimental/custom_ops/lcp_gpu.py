from jax import core, dtypes, lax
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jax.interpreters import xla, ad, batching
from jax.abstract_arrays import ShapedArray
from jbdl.experimental.custom_ops.trace import trace, expectNotImplementedError
from jax.lib import xla_client
#from jbdl.experimental import cpu_ops
from jax.api import jit, vmap
from jax.api import jacfwd, device_put


# If the GPU version exists, also register those
try:
    from jbdl.experimental import gpu_ops
except ImportError:
    gpu_ops = None
else:
    # Register the GPU XLA custom calls
    for name, value in gpu_ops.registrations().items():     #  gpu_lcp_double
        if "lcp" in name:
            xla_client.register_custom_call_target(name, value, platform="gpu")


lcp_gpu_prim = core.Primitive("lcp_gpu")
lcp_gpu_prim.multiple_results = True
lcp_gpu_prim.def_impl(partial(xla.apply_primitive, lcp_gpu_prim))



def lcp_gpu(H, f, L, k, lb, ub):

    H = jax.numpy.triu(H)
    n = L.shape[0]
    m = L.shape[1]
    Im = jax.numpy.eye(m)
    A = jax.numpy.vstack([L, Im])
    lower_bound = jax.numpy.vstack([(-1) * np.inf * jax.numpy.ones([n, 1]), lb])  # (36, 1)
    upper_bound = jax.numpy.vstack([k, ub])

    x, z = lcp_gpu_prim.bind(H, f, A, lower_bound, upper_bound)
    return x, z

# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
# @trace("lcp_abstract_eval")
def lcp_gpu_abstract_eval(P, q, A, l, u):
    n = q.shape[0]
    m = l.shape[0]
    dtype = dtypes.canonicalize_dtype(P.dtype)

    assert dtypes.canonicalize_dtype(q.dtype) == dtype
    assert dtypes.canonicalize_dtype(A.dtype) == dtype
    assert dtypes.canonicalize_dtype(l.dtype) == dtype
    assert dtypes.canonicalize_dtype(u.dtype) == dtype

    assert P.shape == (n, n)
    assert q.shape == (n, 1)
    assert A.shape == (m, n)
    assert l.shape == (m, 1)
    assert u.shape == (m, 1)

    primal_shape = (n, 1)
    dual_shape = (m + 2*n, 1)

    return (ShapedArray(primal_shape, dtype), ShapedArray(dual_shape, dtype))



def lcp_gpu_translation(c, P, q, A, l, u, *, platform="gpu"):
    # The inputs have "shapes" that provide both the shape and the dtype
    P_shape = c.get_shape(P)
    q_shape = c.get_shape(q)
    A_shape = c.get_shape(A)
    l_shape = c.get_shape(l)
    u_shape = c.get_shape(u)

    # Extract the dtype and shape
    dtype = P_shape.element_type()
    n, _ = q_shape.dimensions()
    m, _ = l_shape.dimensions()
    assert q_shape.element_type() == dtype
    assert A_shape.element_type() == dtype
    assert l_shape.element_type() == dtype
    assert u_shape.element_type() == dtype

    assert P_shape.dimensions() == (n, n)
    assert q_shape.dimensions() == (n, 1)
    assert A_shape.dimensions() == (m, n)
    assert l_shape.dimensions() == (m, 1)
    assert u_shape.dimensions() == (m, 1)

    # The input specification
    n_array_shape = xla_client.Shape.array_shape(
        np.dtype(np.int64), (), ())
    m_array_shape = xla_client.Shape.array_shape(
        np.dtype(np.int64), (), ())
    P_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (n, n), (1, 0))
    q_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (n, 1), (1, 0))
    A_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (m, n), (1, 0))
    l_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (m, 1), (1, 0))
    u_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (m, 1), (1, 0))

    # The output specification
    primal_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (n, 1), (1, 0))
    dual_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (m + 2*n, 1), (1, 0))

    op_name = b"gpu_lcp_double"

    if platform == "cpu":
        raise ValueError("You are using the gpu version lcp. Not implemented for platform 'cpu'")

    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'lcp_gpu' module has not been compiled."
            )
        opaque = gpu_ops.build_osqp_descriptor(n, m)
        return xla_client.ops.CustomCallWithLayout(
            c,
            op_name,
            operands=(P, q, A, l, u),
            # The input shapes:
            operand_shapes_with_layout=(
                P_array_shape,
                q_array_shape,
                A_array_shape,
                l_array_shape,
                u_array_shape,
            ),
            shape_with_layout=xla_client.Shape.tuple_shape((primal_array_shape, dual_array_shape)),
            opaque=opaque,
            )
    else:
        raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")




lcp_gpu_prim.def_abstract_eval(lcp_gpu_abstract_eval)
# Connect the XLA translation rules for JIT compilation
xla.backend_specific_translations["gpu"][lcp_gpu_prim] = partial(lcp_gpu_translation, platform="gpu")


def lcp_gpu_kkt(x, z, H, f, L, k, lb, ub):
    n_var = H.shape[1]

    L = jnp.vstack([L, -np.eye(n_var)])
    k = jnp.vstack([k, -lb])
    L = jnp.vstack([L, np.eye(n_var)])
    k = jnp.vstack([k, ub])

    lagrange = jnp.matmul(H, x) + f + jnp.matmul(jnp.transpose(L), z)
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
def lcp_gpu_jvp(arg_values, arg_tangents):
    H, f, L, k, lb, ub = arg_values
    H_dot, f_dot, L_dot, k_dot, lb_dot, ub_dot = arg_tangents

    # We use "bind" here because we don't want to mod the mean anomaly again
    x_star, z_star = lcp_gpu(H, f, L, k, lb, ub)
    nV = H.shape[1]

    dkkt2dx = jacfwd(lcp_gpu_kkt, argnums=0)(x_star, z_star, H, f, L, k, lb, ub)
    dkkt2dz = jacfwd(lcp_gpu_kkt, argnums=1)(x_star, z_star, H, f, L, k, lb, ub)

    dkkt2dH = jacfwd(lcp_gpu_kkt, argnums=2)(x_star, z_star, H, f, L, k, lb, ub)
    dkkt2df = jacfwd(lcp_gpu_kkt, argnums=3)(x_star, z_star, H, f, L, k, lb, ub)
    dkkt2dL = jacfwd(lcp_gpu_kkt, argnums=4)(x_star, z_star, H, f, L, k, lb, ub)
    dkkt2dk = jacfwd(lcp_gpu_kkt, argnums=5)(x_star, z_star, H, f, L, k, lb, ub)
    dkkt2dlb = jacfwd(lcp_gpu_kkt, argnums=6)(x_star, z_star, H, f, L, k, lb, ub)
    dkkt2dub = jacfwd(lcp_gpu_kkt, argnums=7)(x_star, z_star, H, f, L, k, lb, ub)

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
    dxz2df = -jnp.linalg.solve(dkkt2dxz, dkkt2df)
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

    if type(H_dot) is ad.Zero:
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

    return (x_star, z_star), (diff[0:nV, ...], diff[nV:, ...])


ad.primitive_jvps[lcp_gpu_prim] = lcp_gpu_jvp
#
#
# # ************************************
# # *  SUPPORT FOR BATCHING WITH VMAP  *
# # ************************************
#
def lcp_gpu_batch(batched_args, batch_dims):
    Hs, fs, Ls, lbs, ubs = batched_args
    Hs_bd, fs_bd, Ls_bd, lbs_bd, ubs_bd = batch_dims

    size = next(t.shape[i] for t, i in zip(batched_args, batch_dims) if i is not None)

    Hs = batching.bdim_at_front(Hs, Hs_bd, size)
    fs = batching.bdim_at_front(fs, fs_bd, size)
    Ls = batching.bdim_at_front(Ls, Ls_bd, size)
    lbs = batching.bdim_at_front(lbs, lbs_bd, size)
    ubs = batching.bdim_at_front(ubs, ubs_bd, size)

    samples = lax.map(lambda args: lcp_gpu_prim.bind(*args), (Hs, fs, Ls, lbs, ubs))

    return samples, (0, 0)

batching.primitive_batchers[lcp_gpu_prim] = lcp_gpu_batch
