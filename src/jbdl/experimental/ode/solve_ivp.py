from functools import partial
from numpy import diff
from jax.api import jacfwd
import jax.numpy  as jnp
from jax._src.util import safe_map, safe_zip
from jax.flatten_util import ravel_pytree
from jax import linear_util as lu
import jax
from jax import lax
from jax import device_put


map = safe_map
zip = safe_zip


def ravel_first_arg(f, unravel):
    return ravel_first_arg_(lu.wrap_init(f), unravel).call_wrapped


@lu.transformation
def ravel_first_arg_(unravel, y_flat, *args):
    y = unravel(y_flat)
    ans = yield (y,) + args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat


def solve_ivp(func, y0, t, event_func, event_handle, *args,  rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf):
    return _solve_ivp_wrapper(func, event_func, event_handle, rtol, atol, mxstep, y0, t, *args)


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
def _solve_ivp_wrapper(func, event_func, event_handle, rtol, atol, mxstep, y0, ts, *args):
    y0, unravel = ravel_pytree(y0)
    func = ravel_first_arg(func, unravel)
    out = _solve_ivp(func, event_func, event_handle, rtol, atol, mxstep, y0, ts, *args)
    return jax.vmap(unravel)(out)


def _solve_ivp(func, event_func, event_handle, rtol, atol, mxstep, y0, ts, *args):

    def func_(y, t):
        return func(y, t, *args)

    def event_func_(y, t):
        return event_func(y, t, *args)

    def event_handle_(y, t):
        return event_handle(y, t, *args)

    def scan_fun(carry, target_t):
        def cond_fun(state):
            i, _, _, t, dt, _, _ = state
            return (t < target_t) & (i < mxstep) & (dt > 0)

        def body_fun(state):
            i, y, f, t, dt, _, e = state
            next_y = y + f * dt
            next_t = t + dt
            next_f = func_(next_y, next_t)
            next_e = event_func_(next_y, next_t)
            event_handle_y = event_handle_(next_y, next_t)
            event_handle_f = func_(event_handle_y, next_t)
            event_handle_e = event_func_(event_handle_y, next_t)

            new = [i + 1, next_y, next_f, next_t, dt, t, next_e]
            event_handle_new =  [i + 1, event_handle_y, event_handle_f, next_t, dt, t, event_handle_e]

            return map(partial(jnp.where, jnp.all(jnp.logical_and(e > 0, next_e < 0))), event_handle_new, new)

        _, *carry = lax.while_loop(cond_fun, body_fun, [0] + carry)
        y, _, _, _, _, _ = carry

        return carry, y

    f0 = func_(y0, ts[0])
    dt = device_put(0.001)
    e0 = device_put(1.0)

    init_carry = [y0, f0, ts[0], dt, ts[0], e0]

    _, ys = lax.scan(scan_fun, init_carry, ts[1:])

    return jnp.concatenate((y0[None], ys))



if __name__ == "__main__":
    print("Hello!")
    import time

    def e_handle(y, t, *args):
        return -y


    def e_fun(y, t, *args):
        return y[0]

    def pend(y, t, b, c):
        dxdt = jnp.array([y[1], -b*y[1] - c*jnp.sin(y[0])])
        return dxdt

    y0 = jnp.array([jnp.pi - 0.1, 0.0])
    B = 0.25
    C = 5.0

    t_eval =  jnp.linspace(0, 10, 1000)

    sol = jax.jit(solve_ivp, static_argnums=(0, 3, 4))(pend, y0, t_eval, e_fun, e_handle, B, C)
    print(sol)

    print("------------------")
    start = time.time()
    result = solve_ivp(pend, y0, jnp.linspace(0, 1, 1000), e_fun, e_handle, B, C)
    result.block_until_ready()
    duration = time.time() - start

    print(duration)

    start = time.time()
    result = solve_ivp(pend, y0, jnp.linspace(0, 1, 1000), e_fun, e_handle, B, C)
    result.block_until_ready()
    duration = time.time() - start

    print(duration)

    print("=================")

    # pure_solve_ivp = partial(solve_ivp, func=pend, event_fun=e_fun, event_handle=e_handle)


    start = time.time()
    diff = jax.jit(jacfwd(solve_ivp, argnums=1), static_argnums=(0, 3, 4))
    reslut = diff(pend, y0, jnp.linspace(0, 1, 1000), e_fun, e_handle, B, C)
    result.block_until_ready()
    duration = time.time() - start
    print(duration)


    start = time.time()
    reslut = diff(pend, y0, jnp.linspace(0, 1, 1000), e_fun, e_handle, B, C)
    result.block_until_ready()
    duration = time.time() - start
    print(duration)

    start = time.time()
    reslut = diff(pend, y0, jnp.linspace(0, 1, 1000), e_fun, e_handle, B, C)
    result.block_until_ready()
    duration = time.time() - start
    print(duration)
