from jax.api import jacrev
from jax.experimental.ode import odeint
from numpy.core.numeric import argwhere
from scipy.integrate.quadpack import tplquad
import jax.numpy as jnp

def integrate_dynamics(func, y0, t_span, de_righta_t, event=None, args=None, rtol=1e-5, atol=1e-5, mxstep=jnp.inf):
    t0, tf = t_span

    # lists to store the results
    t_eval = [t0]
    sol = [y0]
    if event is not None:
        e = [event(y0, t0)]
        events = []
    t2 = t0

    while t2 < tf: 
        t1 = t_eval[-1]
        t2 = t1 + de_righta_t
        if t2 > tf - atol:
            t2 = tf
        y1 = sol[-1]

        if args is not None:      
            y2 = odeint(func, y1, jnp.linspace(t1, t2, 2), *args)
        else:
            y2 = odeint(func, y1, jnp.linspace(t1, t2, 2))


        if event is None:
            t_eval += [t2]
            sol += [y2[-1, :]]
        else:
            # t_eval += [t2]
            # sol += [y2[-1, :]]
            # now evaluate the event at the last position
            next_e = event(y2[-1, :], t2)
            # e += [next_e]
            # print(e)

            if e[-1] > 0 and next_e < 0 :
                # Event detected where the sign of the event has changed from positive to negative. The
                # event is between t_left = X[-2] and t_right = X[-1]. run a modified bisect
                # function to narrow down to find where event = 0
                t_right = t2
                y_right = y2[-1, :]
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
                        print('t = {0}, event = {1}, y = {2}'.format(t_right, e_right, y_right))
                        events += [(t_right, y_right)]

                        t2 = t_right
                        next_e = e_right

                        # print(y1, t1, t2)

                        if args is not None:      
                            y2 = odeint(func, y1, jnp.linspace(t1, t2, 2), *args)
                        else:
                            y2 = odeint(func, y1, jnp.linspace(t1, t2, 2))

                        y2 = -y2
                        break # and return to integrating

                    m = (e_left - e_right)/(t_left - t_right) #slope of line connecting points
                                                #bracketing zero

                    #estimated x where the zero is
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
                sol += [y2[-1, :]]
                e += [next_e]
            else:
                t_eval += [t2]
                sol += [y2[-1, :]]
                e += [next_e]
    
    sol = jnp.vstack(sol)

    return t_eval, sol
        



if __name__ == "__main__":
    import numpy as np
    import jax.numpy as jnp
    
    def pend(y, t, b, c):
        dxdt = jnp.array([y[1], -b*y[1] - c*jnp.sin(y[0])])
        return dxdt

    y0 = np.array([np.pi - 0.1, 0.0])
    b = 0.25
    c = 5.0
    
    t_span = (0, 10)
    de_righta_t = 0.1
    t_num = 102

    def event(y, t):
        return y[0] 

    event = None

    # t_eval, sol = integrate_dynamics(pend, y0, t_span, de_righta_t, event=event, args=(b, c), rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf)

    # print(t_eval)
    # print(len(t_eval))
    # print(sol.shape)
    # print(sol)

    def forward(y0, t_span, de_righta_t, event, b, c):
        t_eval, sol =  integrate_dynamics(pend, y0, t_span, de_righta_t, event, args=(b, c))
        yT = sol[-1, :]
        return yT

    def forward_all(y0, t_span, de_righta_t, event, b, c):
        t_eval, sol =  integrate_dynamics(pend, y0, t_span, de_righta_t, event, args=(b, c))
        return t_eval, sol




    # def dynamics_fwd(y0, tspan, tnum, b, c):
    #     t = jnp.linspace(tspan[0], tnum, tspan[1])
    #     sol = odeint(pend, y0, t, b, c)
    #     yT = sol[-1, :]
    #     return yT

    # def dynamics_fwd_path(y0, tspan, tnum, b, c):
    #     t = jnp.linspace(tspan[0], tnum, tspan[1])
    #     sol = odeint(pend, y0, t, b, c)
    #     return sol

    # sol = dynamics_fwd_path(y0, t_span, t_num, b, c)

    t_eval, sol = forward_all(y0, t_span, de_righta_t, event, b, c)
    # print(t_eval)
    print(len(t_eval))
    print(sol.shape)

    # import matplotlib.pyplot as plt
    # plt.plot(t_eval, sol[:, 0], 'b', label='theta(t)')
    # plt.plot(t_eval, sol[:, 1], 'g', label='omega(t)')
    # plt.legend(loc='best')
    # plt.xlabel('t')
    # plt.grid()
    # plt.show()

    # print(forward(y0, t_span, de_righta_t, event, b, c))


    print(jacrev(forward, argnums=[4, 5])(y0, t_span, de_righta_t, event, b, c))


    # print(jacrev(dynamics_fwd, argnums=[3, 4])(y0, tspan, tnum, b, c))
    
