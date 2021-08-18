from functools import partial
import numpy as np
from jax import core, dtypes, lax
import jax.numpy as jnp
from jax.interpreters import xla, ad, batching
from jax.abstract_arrays import ShapedArray
from jax.lib import xla_client
from jax.api import jit, vmap
from jax.api import jacfwd, device_put
from jax.lib import xla_bridge
from jbdl.experimental import cpu_ops


# Register the CPU XLA custom calls
for name, value in cpu_ops.registrations().items():
    if "lcp" in name:
        xla_client.register_cpu_custom_call_target(name, value)

lcp_prim = core.Primitive("lcp")
lcp_prim.multiple_results = True
lcp_prim.def_impl(partial(xla.apply_primitive, lcp_prim))


def lcp(hh, f, ll, k, lb, ub):

    x, z = lcp_prim.bind(hh, f, ll, k, lb, ub)
    return x, z


def lcp_abstract_eval(hh, f, ll, k, lb, ub):
    nv = f.shape[0]
    nc = k.shape[0]
    dtype = dtypes.canonicalize_dtype(hh.dtype)
    assert dtypes.canonicalize_dtype(f.dtype) == dtype
    assert dtypes.canonicalize_dtype(ll.dtype) == dtype
    assert dtypes.canonicalize_dtype(k.dtype) == dtype
    assert dtypes.canonicalize_dtype(lb.dtype) == dtype
    assert dtypes.canonicalize_dtype(ub.dtype) == dtype
    assert hh.shape == (nv, nv)
    assert f.shape == (nv, 1)
    assert ll.shape == (nc, nv)
    assert k.shape == (nc, 1)
    assert lb.shape == (nv, 1)
    assert ub.shape == (nv, 1)
    primal_shape = (nv, 1)
    dual_shape = (nc + 2 * nv, 1)
    return (ShapedArray(primal_shape, dtype), ShapedArray(dual_shape, dtype))


def cpu_lcp_translation(c, hh, f, ll, k, lb, ub):
    # The inputs have "shapes" that provide both the shape and the dtype
    hh_shape = c.get_shape(hh)
    f_shape = c.get_shape(f)
    ll_shape = c.get_shape(ll)
    k_shape = c.get_shape(k)
    lb_shape = c.get_shape(lb)
    ub_shape = c.get_shape(ub)

    # Extract the dtype and shape
    dtype = hh_shape.element_type()
    nv, _ = f_shape.dimensions()
    nc, _ = k_shape.dimensions()
    assert f_shape.element_type() == dtype
    assert ll_shape.element_type() == dtype
    assert k_shape.element_type() == dtype
    assert lb_shape.element_type() == dtype
    assert ub_shape.element_type() == dtype
    assert hh_shape.dimensions() == (nv, nv)
    assert f_shape.dimensions() == (nv, 1)
    assert ll_shape.dimensions() == (nc, nv)
    assert k_shape.dimensions() == (nc, 1)
    assert lb_shape.dimensions() == (nv, 1)
    assert ub_shape.dimensions() == (nv, 1)


    # The input specification
    nv_array_shape = xla_client.Shape.array_shape(
        np.dtype(np.int64), (), ())
    nc_array_shape = xla_client.Shape.array_shape(
        np.dtype(np.int64), (), ())
    hh_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nv, nv), (1, 0))
    f_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nv, 1), (1, 0))
    ll_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nc, nv), (1, 0))
    k_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nc, 1), (1, 0))
    lb_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nv, 1), (1, 0))
    ub_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nv, 1), (1, 0))

    # The output specification
    primal_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nv, 1), (1, 0))
    dual_array_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), (nc + 2 * nv, 1), (1, 0))


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
        operands=(xla_client.ops.ConstantLiteral(c, nv), xla_client.ops.ConstantLiteral(c, nc), hh, f, ll, k, lb, ub),
        # The input shapes:
        operand_shapes_with_layout=(
            nv_array_shape,
            nc_array_shape,
            hh_array_shape,
            f_array_shape,
            ll_array_shape,
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


def lcp_kkt(x, z,  hh, f, ll, k, lb, ub):
    n_var = hh.shape[1]

    ll = jnp.vstack([ll, -np.eye(n_var)])
    k = jnp.vstack([k, -lb])
    ll = jnp.vstack([ll, np.eye(n_var)])
    k = jnp.vstack([k, ub])

    lagrange = jnp.matmul(hh, x) + f + jnp.matmul(jnp.transpose(ll),  z)
    inequality = (jnp.matmul(ll, x) - k) * z
    kkt = jnp.vstack([lagrange, inequality])
    return kkt


def lcp_jvp(arg_values, arg_tangents):
    hh, f, ll, k, lb, ub = arg_values
    hh_dot, f_dot, ll_dot, k_dot, lb_dot, ub_dot = arg_tangents

    x_star, z_star = lcp_prim.bind(hh, f, ll, k, lb, ub)
    nv = hh.shape[1]

    dkkt2dx = jacfwd(lcp_kkt, argnums=0)(x_star, z_star,  hh, f, ll, k, lb, ub)
    dkkt2dz = jacfwd(lcp_kkt, argnums=1)(x_star, z_star,  hh, f, ll, k, lb, ub)
    dkkt2dhh = jacfwd(lcp_kkt, argnums=2)(x_star, z_star,  hh, f, ll, k, lb, ub)
    dkkt2df = jacfwd(lcp_kkt, argnums=3)(x_star, z_star,  hh, f, ll, k, lb, ub)
    dkkt2dll = jacfwd(lcp_kkt, argnums=4)(x_star, z_star,  hh, f, ll, k, lb, ub)
    dkkt2dk = jacfwd(lcp_kkt, argnums=5)(x_star, z_star,  hh, f, ll, k, lb, ub)
    dkkt2dlb = jacfwd(lcp_kkt, argnums=6)(x_star, z_star,  hh, f, ll, k, lb, ub)
    dkkt2dub = jacfwd(lcp_kkt, argnums=7)(x_star, z_star,  hh, f, ll, k, lb, ub)
    dkkt2dxz = jnp.concatenate([dkkt2dx, dkkt2dz], axis=2)
    dkkt2dxz = jnp.transpose(dkkt2dxz, [3, 1, 0, 2])
    dkkt2dhh = jnp.transpose(dkkt2dhh, [2, 3, 0, 1])
    dkkt2df = jnp.transpose(dkkt2df, [2, 3, 0, 1])
    dkkt2dll = jnp.transpose(dkkt2dll, [2, 3, 0, 1])
    dkkt2dk = jnp.transpose(dkkt2dk, [2, 3, 0, 1])
    dkkt2dlb = jnp.transpose(dkkt2dlb, [2, 3, 0, 1])
    dkkt2dub = jnp.transpose(dkkt2dub, [2, 3, 0, 1])
    dxz2dhh = -jnp.linalg.solve(dkkt2dxz, dkkt2dhh)
    dxz2dhh = jnp.transpose(dxz2dhh, [2, 3, 0, 1])
    dxz2df =  -jnp.linalg.solve(dkkt2dxz, dkkt2df)
    dxz2df = jnp.transpose(dxz2df, [2, 3, 0, 1])
    dxz2dll = -jnp.linalg.solve(dkkt2dxz, dkkt2dll)
    dxz2dll = jnp.transpose(dxz2dll, [2, 3, 0, 1])
    dxz2dk = -jnp.linalg.solve(dkkt2dxz, dkkt2dk)
    dxz2dk = jnp.transpose(dxz2dk, [2, 3, 0, 1])
    dxz2dlb = -jnp.linalg.solve(dkkt2dxz, dkkt2dlb)
    dxz2dlb = jnp.transpose(dxz2dlb, [2, 3, 0, 1])
    dxz2dub = -jnp.linalg.solve(dkkt2dxz, dkkt2dub)
    dxz2dub = jnp.transpose(dxz2dub, [2, 3, 0, 1])

    if type(hh_dot) is  ad.Zero:
        diff_hh = device_put(0.0)
    else:
        diff_hh = jnp.sum(dxz2dhh * hh_dot, axis=(-2, -1))

    if type(f_dot) is ad.Zero:
        diff_f = device_put(0.0)
    else:
        diff_f = jnp.sum(dxz2df * f_dot, axis=(-2, -1))

    if type(ll_dot) is ad.Zero:
        diff_ll = device_put(0.0)
    else:
        diff_ll = jnp.sum(dxz2dll * ll_dot, axis=(-2, -1))

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

    diff = diff_hh + diff_f + diff_ll + diff_k + diff_lb + diff_ub

    return (x_star, z_star), (diff[0:nv, ...], diff[nv:, ...])

ad.primitive_jvps[lcp_prim] = lcp_jvp


def lcp_batch(batched_args, batch_dims):
    hhs, fs, lls, ks, lbs, ubs = batched_args
    hhs_bd, fs_bd, lls_bd, ks_bd, lbs_bd, ubs_bd = batch_dims

    size = next(t.shape[i] for t, i in zip(batched_args, batch_dims) if i is not None)

    hhs = batching.bdim_at_front(hhs, hhs_bd, size)
    fs = batching.bdim_at_front(fs, fs_bd, size)
    lls = batching.bdim_at_front(lls, lls_bd, size)
    ks = batching.bdim_at_front(ks, ks_bd, size)
    lbs = batching.bdim_at_front(lbs, lbs_bd, size)
    ubs = batching.bdim_at_front(ubs, ubs_bd, size)

    if xla_bridge.get_backend().platform == 'cpu':
        samples = lax.map(lambda args: lcp(*args), (hhs, fs, lls, ks, lbs, ubs))
    else:
        raise NotImplementedError(f"Unsupported platform")
    return samples, (0, 0)


batching.primitive_batchers[lcp_prim] = lcp_batch


if __name__ == "__main__":
    from jax.config import config
    config.update("jax_enable_x64", True)
    hh = np.array([1.0, -1.0, -1.0, 2.0]).reshape(2, 2)
    f = np.array([-2.0, -6.0]).reshape(2, 1)
    ll = np.array([1.0, 1.0, -1.0, 2.0, 2.0, 1.0]).reshape(3, 2)
    k = np.array([2.0, 2.0, 3.0]).reshape(3, 1)
    lb = np.array([0.0, 0.0]).reshape(2, 1)
    ub = np.array([0.5, 5.0 ]).reshape(2, 1)
    print(jit(lcp)(hh, f, ll, k, lb, ub))

    fwd = jacfwd(lcp, argnums=2)(hh, f, ll, k, lb, ub)
    print(fwd)

    BATCH_SIZE = 10

    batch_hh = jnp.repeat(jnp.expand_dims(hh, axis=0), BATCH_SIZE, axis=0)
    batch_f = jnp.repeat(jnp.expand_dims(f, axis=0), BATCH_SIZE, axis=0)
    batch_ll = jnp.repeat(jnp.expand_dims(ll, axis=0), BATCH_SIZE, axis=0)
    batch_k = jnp.repeat(jnp.expand_dims(k, axis=0), BATCH_SIZE, axis=0)
    batch_lb = jnp.repeat(jnp.expand_dims(lb, axis=0), BATCH_SIZE, axis=0)
    batch_ub = jnp.repeat(jnp.expand_dims(ub, axis=0), BATCH_SIZE, axis=0)
    print(batch_hh.shape)
    print(batch_f.shape)
    print(batch_ll.shape)
    print(batch_k.shape)
    print(batch_lb.shape)
    print(batch_ub.shape)
    batch_x, batch_z = vmap(lcp, 0)(batch_hh, batch_f, batch_ll, batch_k, batch_lb, batch_ub)
    print(batch_x)
    print(batch_x.shape)

    print(batch_z)
    print(batch_z.shape)
