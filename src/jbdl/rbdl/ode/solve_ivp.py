from jax.api import jacrev
from jax.experimental.ode import odeint
from numpy.core.numeric import argwhere
from scipy.integrate.quadpack import tplquad
import jax.numpy as jnp

def integrate_dynamics(func, y0, t_span, delta_t, event=None, args=None, rtol=1e-5, atol=1e-5, mxstep=jnp.inf):
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
        t2 = t1 + delta_t
        if t2 > tf - atol:
            t2 = tf
        y1 = sol[-1]

        if args is not None:      
            y2 = odeint(func, y1, jnp.linspace(t1, t2, 2), *args)
        else:
            y2 = odeint(func, y1, jnp.linspace(t1, t2, 2))
        t_eval += [t2]
        sol += [y2[-1, :]]
        
        if event is not None:
            # now evaluate the event at the last position
            e += [event(sol[-1], t_eval[-1])]
            # print(e)

            if e[-2] > 0 and e[-1] < 0 :
                # Event detected where the sign of the event has changed from positive to negative. The
                # event is between xPt = X[-2] and xLt = X[-1]. run a modified bisect
                # function to narrow down to find where event = 0
                xLt = t_eval[-1]
                fLt = sol[-1]
                eLt = e[-1]

                xPt = t_eval[-2]
                fPt = sol[-2]
                ePt = e[-2]

                j = 0
                while j < 100:
                    if jnp.abs(xLt - xPt) < 1e-4:
                        # we know the interval to a prescribed precision now.
                        # print 'Event found between {0} and {1}'.format(x1t, x2t)
                        print('t = {0}, event = {1}, y = {2}'.format(xLt, eLt, fLt))
                        events += [(xLt, fLt)]

                        # Deal with event.
                        # t_eval[-1] = xLt
                        # sol[-1] = -fLt

                        break # and return to integrating

                    m = (ePt - eLt)/(xPt - xLt) #slope of line connecting points
                                                #bracketing zero

                    #estimated x where the zero is
                    inc_xPt = - ePt / m
                    const_inc_xPt = 0.618 * (xLt - xPt)

                    if inc_xPt > const_inc_xPt:     
                        new_x = const_inc_xPt + xPt
                    else:
                        new_x =  inc_xPt + xPt
                    

                    # now get the new value of the integrated solution at that new x
                    if args is not None:
                        f  = odeint(func, fPt, jnp.linspace(xPt, new_x, 2), *args)
                    else:
                        f  = odeint(func, fPt, jnp.linspace(xPt, new_x, 2))
                    
                    new_f = f[-1, :]
                    
                    new_e = event(new_f, new_x)

                    if new_e > 0:
                        xPt = new_x
                        fPt = new_f
                        ePt = new_e
                    else:
                        xLt = new_x
                        fLt = new_f
                        eLt = new_e

                    j += 1
    
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
    delta_t = 0.1
    t_num = 102

    def event(y, t):
        return y[0] 

    # event = None

    # t_eval, sol = integrate_dynamics(pend, y0, t_span, delta_t, event=event, args=(b, c), rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf)

    # print(t_eval)
    # print(len(t_eval))
    # print(sol.shape)
    # print(sol)

    def forward(y0, t_span, delta_t, event, b, c):
        t_eval, sol =  integrate_dynamics(pend, y0, t_span, delta_t, event, args=(b, c))
        yT = sol[-1, :]
        return yT

    def forward_all(y0, t_span, delta_t, event, b, c):
        t_eval, sol =  integrate_dynamics(pend, y0, t_span, delta_t, event, args=(b, c))
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

    t_eval, sol = forward_all(y0, t_span, delta_t, event, b, c)
    print(len(t_eval))
    print(sol.shape)

    # import matplotlib.pyplot as plt
    # plt.plot(t_eval, sol[:, 0], 'b', label='theta(t)')
    # plt.plot(t_eval, sol[:, 1], 'g', label='omega(t)')
    # plt.legend(loc='best')
    # plt.xlabel('t')
    # plt.grid()
    # plt.show()

    # print(forward(y0, t_span, delta_t, event, b, c))


    print(jacrev(forward, argnums=[4, 5])(y0, t_span, delta_t, event, b, c))


    # print(jacrev(dynamics_fwd, argnums=[3, 4])(y0, tspan, tnum, b, c))
    
