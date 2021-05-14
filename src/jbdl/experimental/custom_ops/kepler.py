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


# Register the CPU XLA custom calls

for name, value in cpu_ops.registrations().items():
    if "kepler" in name:
        xla_client.register_cpu_custom_call_target(name, value)


kepler_prim = core.Primitive("kepler")
kepler_prim.multiple_results = True
kepler_prim.def_impl(partial(xla.apply_primitive, kepler_prim))

# This function exposes the primitive to user code and this is the only
# public-facing function in this module
# @trace("kepler")
def kepler(mean_anom, ecc):
    # We're going to apply array broadcasting here since the logic of our op
    # is much simpler if we require the inputs to all have the same shapes
    mean_anom_, ecc_ = jnp.broadcast_arrays(mean_anom, ecc)

    # Then we need to wrap into the range [0, 2*pi)
    M_mod = jnp.mod(mean_anom_, 2 * np.pi)

    return kepler_prim.bind(M_mod, ecc_)



# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
# @trace("kelper_abstract_eval")
def kepler_abstract_eval(mean_anom, ecc):
    shape = mean_anom.shape
    dtype = dtypes.canonicalize_dtype(mean_anom.dtype)
    assert dtypes.canonicalize_dtype(ecc.dtype) == dtype
    assert ecc.shape == shape
    return (ShapedArray(shape, dtype), ShapedArray(shape, dtype))


# We also need a translation rule to convert the function into an XLA op. In
# our case this is the custom XLA op that we've written. We're wrapping two
# translation rules into one here: one for the CPU and one for the GPU
# @trace("kepler_translation")
def kepler_translation(c, mean_anom, ecc, *, platform="cpu"):
    # The inputs have "shapes" that provide both the shape and the dtype
    mean_anom_shape = c.get_shape(mean_anom)
    ecc_shape = c.get_shape(ecc)

    # Extract the dtype and shape
    dtype = mean_anom_shape.element_type()
    dims = mean_anom_shape.dimensions()
    assert ecc_shape.element_type() == dtype
    assert ecc_shape.dimensions() == dims

    # The total size of the input is the product across dimensions
    size = np.prod(dims).astype(np.int64)

    # The inputs and outputs all have the same shape so let's predefine this
    # specification
    shape = xla_client.Shape.array_shape(
        np.dtype(dtype), dims, tuple(range(len(dims) - 1, -1, -1))
    )

    # We dispatch a different call depending on the dtype
    if dtype == np.float32:
        op_name = platform.encode() + b"_kepler_f32"
    elif dtype == np.float64:
        op_name = platform.encode() + b"_kepler_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we pass the size of the data as a the first input
        # argument
        return xla_client.ops.CustomCallWithLayout(
            c,
            op_name,
            # The inputs:
            operands=(xla_client.ops.ConstantLiteral(c, size), mean_anom, ecc),
            # The input shapes:
            operand_shapes_with_layout=(
                xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
                shape,
                shape,
            ),
            # The output shapes:
            shape_with_layout=xla_client.Shape.tuple_shape((shape, shape)),
        )

    elif platform == "gpu":
        raise ValueError("Not implemented for 'gpu'")
        # if gpu_ops is None:
        #     raise ValueError(
        #         "The 'kepler_jax' module was not compiled with CUDA support"
        #     )

        # # On the GPU, we do things a little differently and encapsulate the
        # # dimension using the 'opaque' parameter
        # opaque = gpu_ops.build_kepler_descriptor(size)

        # return xla_client.ops.CustomCallWithLayout(
        #     c,
        #     op_name,
        #     operands=(mean_anom, ecc),
        #     operand_shapes_with_layout=(shape, shape),
        #     shape_with_layout=xla_client.Shape.tuple_shape((shape, shape)),
        #     opaque=opaque,
        # )

    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")

kepler_prim.def_abstract_eval(kepler_abstract_eval)
# Connect the XLA translation rules for JIT compilation
xla.backend_specific_translations["cpu"][kepler_prim] = partial(kepler_translation, platform="cpu")


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************

# Here we define the differentiation rules using a JVP derived using implicit
# differentiation of Kepler's equation:
#
#  M = E - e * sin(E)
#  -> dM = dE * (1 - e * cos(E)) - de * sin(E)
#  -> dE/dM = 1 / (1 - e * cos(E))  and  de/dM = sin(E) / (1 - e * cos(E))
#
# In this case we don't need to define a transpose rule in order to support
# reverse and higher order differentiation. This might not be true in other
# applications, so check out the "How JAX primitives work" tutorial in the JAX
# documentation for more info as necessary.
def kepler_jvp(args, tangents):
    mean_anom, ecc = args
    d_mean_anom, d_ecc = tangents

    # We use "bind" here because we don't want to mod the mean anomaly again
    sin_ecc_anom, cos_ecc_anom = kepler_prim.bind(mean_anom, ecc)

    def zero_tangent(tan, val):
        return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan

    # Propagate the derivatives
    d_ecc_anom = (
        zero_tangent(d_mean_anom, mean_anom)
        + zero_tangent(d_ecc, ecc) * sin_ecc_anom
    ) / (1 - ecc * cos_ecc_anom)

    return (sin_ecc_anom, cos_ecc_anom), (
        cos_ecc_anom * d_ecc_anom,
        -sin_ecc_anom * d_ecc_anom,
    )

# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************

# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.

def kepler_batch(args, axes):
    assert axes[0] == axes[1]
    return kepler(*args), axes


batching.primitive_batchers[kepler_prim] = kepler_batch







# kepler_prim.def_impl(partial(xla.apply_primitive, _kepler_prim))
# _kepler_prim.def_abstract_eval(_kepler_abstract)

# # Connect the XLA translation rules for JIT compilation
# xla.backend_specific_translations["cpu"][_kepler_prim] = partial(
#     _kepler_translation, platform="cpu"
# )
# xla.backend_specific_translations["gpu"][_kepler_prim] = partial(
#     _kepler_translation, platform="gpu"
# )

# # Connect the JVP and batching rules
# ad.primitive_jvps[_kepler_prim] = _kepler_jvp
# batching.primitive_batchers[_kepler_prim] = _kepler_batch

if __name__ == "__main__":
    # with expectNotImplementedError():
    
    print(kepler([2., 3.0], [10., 3.0]))
    print(jit(lambda x, y: kepler(x, y))(2., 10.) )

    # print(kepler(2.0, 10.0))