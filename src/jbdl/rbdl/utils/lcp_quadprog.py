from jax import core
import functools
import traceback
import jax
import numpy as np
import jax.numpy as jnp
from numpy.lib.utils import deprecate
from jbdl.rbdl.contact.solve_contact_lcp import cvxopt_quadprog
from jax import abstract_arrays
from jax.interpreters import ad
from jax.api import device_put, jacfwd, jacrev


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


lcp_quadprog_p = core.Primitive("lcp_quadprog")
lcp_p = core.Primitive("lcp")


def lcp_quadprog_prim(H, f, L, k, lb, ub):
    """The JAX-traceable way to use the JAX primitive.
    
    Note that the traced arguments must be passed as positional arguments
    to `bind`. 
    """
    return lcp_quadprog_p.bind(H, f, L, k, lb, ub)


def lcp_prim(H, f, L, k, lb, ub):
    """The JAX-traceable way to use the JAX primitive.
    
    Note that the traced arguments must be passed as positional arguments
    to `bind`. 
    """
    return lcp_p.bind(H, f, L, k, lb, ub)


def lcp_quadprog_impl(H, f, L, k, lb, ub):
    """Concrete implementation of the primitive.

    This function does not need to be JAX traceable.
    Args:
        H, f, L, k, lb, ub: the concrete arguments of the primitive. Will only be called with 
        concrete values.
    Returns:
        the concrete result of the primitive.
    """
    # Note that we can use the original numpy, which is not JAX traceable
    cvxopt_qp_H = np.asfarray(H)
    cvxopt_qp_f = np.asfarray(f)
    cvxopt_qp_L = np.asfarray(L)
    cvxopt_qp_k = np.asfarray(k)
    cvxopt_qp_lb = np.asfarray(lb)
    cvxopt_qp_ub = np.asfarray(ub)

    x, _, z, status = cvxopt_quadprog(cvxopt_qp_H, cvxopt_qp_f, L=cvxopt_qp_L, k=cvxopt_qp_k, lb=cvxopt_qp_lb, ub=cvxopt_qp_ub)
    if status != 'optimal':
        print('QP solve failed: status = %', status)
    return x, z


def lcp_impl(H, f, L, k, lb, ub):
    """Concrete implementation of the primitive.

    This function does not need to be JAX traceable.
    Args:
        H, f, L, k, lb, ub: the concrete arguments of the primitive. Will only be called with 
        concrete values.
    Returns:
        the concrete result of the primitive.
    """
    x, _ = lcp_quadprog_prim(H, f, L, k, lb, ub)
    return x

lcp_quadprog_p.def_impl(lcp_quadprog_impl)
lcp_p.def_impl(lcp_impl)


def lcp_quadprog_abstract_eval(Hs, fs, Ls, ks, lbs, ubs):
    """Abstract evaluation of the primitive.

    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
        xs, ys, zs: abstractions of the arguments.
    Result:
        a ShapedArray for the result of the primitive.
    """
    zs_shape = (ks.shape[0] + lbs.shape[0] + ubs.shape[0], 1)
    return (abstract_arrays.ShapedArray(fs.shape, fs.dtype), abstract_arrays.ShapedArray(zs_shape, ks.dtype))


def lcp_abstract_eval(Hs, fs, Ls, ks, lbs, ubs):
    """Abstract evaluation of the primitive.

    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
        xs, ys, zs: abstractions of the arguments.
    Result:
        a ShapedArray for the result of the primitive.
    """
   
    return abstract_arrays.ShapedArray(fs.shape, fs.dtype)

# Now we register the abstract evaluation with JAX
lcp_quadprog_p.def_abstract_eval(lcp_quadprog_abstract_eval)
lcp_p.def_abstract_eval(lcp_abstract_eval)


def lcp_value_and_jvp(arg_values, arg_tangents):
    """Evaluates the primal output and the tangents (Jacobian-vector product).

    Given values of the arguments and perturbation of the arguments (tangents), 
    compute the output of the primitive and the perturbation of the output.

    This method must be JAX-traceable. JAX may invoke it with abstract values 
    for the arguments and tangents.

    Args:
        arg_values: a tuple of arguments
        arg_tangents: a tuple with the tangents of the arguments. The tuple has 
        the same length as the arg_values. Some of the tangents may also be the 
        special value ad.Zero to specify a zero tangent.
    Returns:
        a pair of the primal output and the tangent.
    """
    H, f, L, k, lb, ub = arg_values
    H_dot, f_dot, L_dot, k_dot, lb_dot, ub_dot = arg_tangents
 
    # Now we have a JAX-traceable computation of the output. 
    # Normally, we can use the ma primtive itself to compute the primal output. 
    x_star, z_star = lcp_quadprog_prim(H, f, L, k, lb, ub)

    n_var = H.shape[1]
    # We must use a JAX-traceable way to compute the tangent. It turns out that 
    # the output tangent can be computed as (xt * y + x * yt + zt),
    # which we can implement in a JAX-traceable way using the same "multiply_add_prim" primitive.
    
    # We do need to deal specially with Zero. Here we just turn it into a 
    # proper tensor of 0s (of the same shape as 'x'). 
    # An alternative would be to check for Zero and perform algebraic 
    # simplification of the output tangent computation.
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
    dxz2dH = dxz2dH[0:n_var, ...]
    dxz2df =  -jnp.linalg.solve(dkkt2dxz, dkkt2df)
    dxz2df = jnp.transpose(dxz2df, [2, 3, 0, 1])
    dxz2df = dxz2df[0:n_var, ...]
    dxz2dL = -jnp.linalg.solve(dkkt2dxz, dkkt2dL)
    dxz2dL = jnp.transpose(dxz2dL, [2, 3, 0, 1])
    dxz2dL = dxz2dL[0:n_var, ...]
    dxz2dk = -jnp.linalg.solve(dkkt2dxz, dkkt2dk)
    dxz2dk = jnp.transpose(dxz2dk, [2, 3, 0, 1])
    dxz2dk = dxz2dk[0:n_var, ...]
    dxz2dlb = -jnp.linalg.solve(dkkt2dxz, dkkt2dlb)
    dxz2dlb = jnp.transpose(dxz2dlb, [2, 3, 0, 1])
    dxz2dlb = dxz2dlb[0:n_var, ...]
    dxz2dub = -jnp.linalg.solve(dkkt2dxz, dkkt2dub)
    dxz2dub = jnp.transpose(dxz2dub, [2, 3, 0, 1])
    dxz2dub = dxz2dub[0:n_var, ...]


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

    return x_star, diff_H + diff_f + diff_L + diff_k + diff_lb + diff_ub

# Register the forward differentiation rule with JAX 
ad.primitive_jvps[lcp_p] = lcp_value_and_jvp


if __name__ == "__main__":
    H = jnp.array([[1.0, -1.0],
                   [-1.0, 2.0]])
    f = jnp.array([[-2.0], [-6.0]])
    L = jnp.array([[1.0, 1.0],
                    [-1.0, 2.0], 
                    [2.0, 1.0]])
    k = jnp.array([[2.0], [2.0], [3.0]])

    lb = jnp.array([[0.0], [0.0]])
    ub = jnp.array([[0.5], [5.0]])
    with np.printoptions(precision=2, suppress=True, threshold=5):
        print(lcp_prim(H, f, L, k, lb, ub))
        primal, dual = lcp_quadprog_impl(H, f, L, k, lb, ub)
        print(primal)
        for i in dual:
            print(i)
    # dx2dk = jacfwd(lcp_prim, argnums=3)(H, f, L, k, lb, ub)
    # print("+++++++")
    # print(dx2dk)
    # print(dx2dk.shape)
    # print("----------")
    # print(jacrev(lcp_prim, argnums=3)(H, f, L, k, lb, ub))
    # with expectNotImplementedError():
    #     lcp_prim(H, f, L, k, lb, ub)



