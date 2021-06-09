from functools import partial
from jax.api import jacfwd, jacrev
# from jax.experimental.ode import odeint
from jbdl.experimental.ode.runge_kutta import odeint
from numpy.core.numeric import argwhere
from scipy.integrate.quadpack import tplquad
import jax.numpy as jnp
from jax import device_put

def integrate_dynamics(func, y0, t_span, delta_t, event=None, e_fun=None, args=None, rtol=1e-5, atol=1e-5, mxstep=jnp.inf):
    t0, tf = t_span
    t0 = device_put(t0)
    tf = device_put(tf)

    # lists to store the results
    t_eval = [t0]
    sol = [y0]
    if event is not None:
        e = [event(y0, t0, *args)]
        events = []
    t2 = t0

    while t2 < tf: 
        t1 = t_eval[-1]
        t2 = t1 + delta_t
        if t2 > tf - atol:
            t2 = tf
        y1 = sol[-1]

        if args is not None:      
            y12 = odeint(func, y1, jnp.linspace(t1, t2, 2), *args)
        else:
            y12 = odeint(func, y1, jnp.linspace(t1, t2, 2))
        y2 = y12[-1, :]

        if event is None:
            t_eval += [t2]
            sol += [y2]
        else:
            # t_eval += [t2]
            # sol += [y2[-1, :]]
            # now evaluate the event at the last position
            next_e = event(y2, t2, *args)
            # e += [next_e]
            # print(e)

            if e[-1] > 0 and next_e < 0 :
                # Event detected where the sign of the event has changed from positive to negative. The
                # event is between t_left = X[-2] and t_right = X[-1]. run a modified bisect
                # function to narrow down to find where event = 0
                t_right = t2
                y_right = y2
                e_right = next_e

                t_left = t_eval[-1]
                y_left = sol[-1]
                e_left = e[-1]

                j = 0
                while j < 100:
                    # print(j, t_left, t_right)
                    if jnp.abs(e_right) < 1e-3:
                        # we know the interval to a prescribed precision now.
                        # print 'Event found between {0} and {1}'.format(x1t, x2t)
                        # print('t = {0}, event = {1}, y = {2}'.format(t_right, e_right, y_right))
                        events += [(t_right, y_right)]

                        t2 = t_right
                        next_e = e_right

                        # print(y1, t1, t2)

                        if args is not None:      
                            y12 = odeint(func, y1, jnp.linspace(t1, t2, 2), *args)
                        else:
                            y12 = odeint(func, y1, jnp.linspace(t1, t2, 2))

                        if e_fun is not None:
                            y2 = e_fun(y12[-1, :], t2, *args)
                        else:
                            y2 = y12[-1, :]
                        break # and return to integrating
                    #slope of line connecting points bracketing zero
                    m = (e_left - e_right)/(t_left - t_right) 

                    # estimated t where the zero is
                    inc_t_left = - e_left / m
                    const_inc_t_left = 0.618 * (t_right - t_left)

                    if inc_t_left > const_inc_t_left:     
                        new_t = const_inc_t_left + t_left
                    else:
                        new_t =  inc_t_left + t_left
                    

                    # now get the new value of the integrated solution at that new x
                    if args is not None:
                        f  = odeint(func, y_left, jnp.linspace(t_left, new_t, 2), *args)
                    else:
                        f  = odeint(func, y_left, jnp.linspace(t_left, new_t, 2))
                    
                    new_y = f[-1, :]
                    new_e = event(new_y, new_t)


                    if new_e > 0:
                        t_left = new_t
                        y_left = new_y
                        e_left = new_e
                    else:
                        t_right = new_t
                        y_right = new_y
                        e_right = new_e

                    j += 1
        
                t_eval += [t2]
                sol += [y2]
                e += [next_e]
            else:
                t_eval += [t2]
                sol += [y2]
                e += [next_e]
    
    sol = jnp.vstack(sol)

    return t_eval, sol
        



if __name__ == "__main__":
    import numpy as np
    import jax.numpy as jnp
    from jax import jit
    
    def pend(y, t, b, c):
        dxdt = jnp.array([y[1], -b*y[1] - c*jnp.sin(y[0])])
        return dxdt

    y0 = np.array([np.pi - 0.1, 0.0])
    b = 0.25
    c = 5.0
    
    t_span = (0, 10)
    delta_t = 0.1
    t_num = 102

    def event(y, t, *args):
        return y[0] 

    def e_fun(y, t, *args):
        return -y

    # event = None
    # e_fun = None


    def forward(y0, t_span, delta_t, event, e_fun, b, c):
        t_eval, sol =  integrate_dynamics(pend, y0, t_span, delta_t, event, e_fun, args=(b, c))
        yT = sol[-1, :]
        return yT


    def forward_all(y0, t_span, delta_t, event, e_fun, b, c):
        t_eval, sol =  integrate_dynamics(pend, y0, t_span, delta_t, event, e_fun, args=(b, c))
        return t_eval, sol

    # pure_forward_all = partial(forward_all, event=event, e_fun=e_fun)


    t_eval, sol = forward_all(y0, t_span, delta_t, event, e_fun, b, c)

    # print(t_eval)
    # print(len(t_eval))
    # print(sol.shape)

    import matplotlib.pyplot as plt
    plt.plot(t_eval, sol[:, 0], 'b', label='theta(t)')
    plt.plot(t_eval, sol[:, 1], 'g', label='omega(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()
    import time 
    # print("====================")
    # start = time.time()
    # print(jacrev(forward, argnums=[0,])(y0, t_span, delta_t, event, e_fun, b, c))
    # duration = time.time() - start

    # print(duration)

    # start = time.time()
    # print(jacrev(forward, argnums=[0,])(y0, t_span, delta_t, event, e_fun, b, c))
    # duration = time.time() - start

    # print(duration)
    print("------------------")
    start = time.time()
    result = odeint(pend, y0, jnp.linspace(0, 1, 1000), b, c)
    result.block_until_ready()
    duration = time.time() - start

    print(duration)

    start = time.time()
    result = odeint(pend, y0, jnp.linspace(0, 1, 1000), b, c)
    result.block_until_ready()
    duration = time.time() - start

    print(duration)
    print("=================")
    

    start = time.time()
    result = jacrev(odeint, argnums=1)(pend, y0, jnp.linspace(0, 1, 10), b, c)
    result.block_until_ready()
    duration = time.time() - start
    print(duration)
    
    start = time.time()
    result = jacrev(odeint, argnums=1)(pend, y0, jnp.linspace(0, 1, 10), b, c)
    result.block_until_ready()
    duration = time.time() - start
    print(duration)

    start = time.time()
    result = jacrev(odeint, argnums=1)(pend, y0, jnp.linspace(0, 1, 10), b, c)
    result.block_until_ready()
    duration = time.time() - start
    print(duration)

    print("==============")

    # start = time.time()
    # result = jacrev(odeint, argnums=1)(pend, y0, jnp.linspace(0, 1, 10), b, c)
    # result.block_until_ready()
    # duration = time.time() - start
    # print(duration)

    # start = time.time()
    # diff = jit(jacrev(odeint, argnums=1), static_argnums=(0, 3, 4))
    # result = diff(pend, y0, jnp.linspace(0, 1, 10), b, c)
    # result.block_until_ready()
    # duration = time.time() - start
    # print(duration)

    # start = time.time()
    # result = jacrev(odeint, argnums=1)(pend, y0, jnp.linspace(0, 1, 10), b, c)
    # result.block_until_ready()
    # duration = time.time() - start
    # print(duration)

  

    # start = time.time()
    # result = jit(jacrev(odeint, argnums=1), static_argnums=(0,  3, 4))(pend, y0, jnp.linspace(0, 1, 10), b, c)
    # result.block_until_ready()
    # duration = time.time() - start
    # print(duration)


    # start = time.time()
    # result = diff(pend, y0, jnp.linspace(0, 1, 10), b, c)
    # result.block_until_ready()
    # duration = time.time() - start
    # print(duration)

  


    
