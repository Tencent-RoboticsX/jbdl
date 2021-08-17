# %%
import os

import jax
from jbdl.rbdl.utils import ModelWrapper
from jax import device_put
from jbdl.rbdl.utils import xyz2int
import jax.numpy as jnp
from jbdl.rbdl.dynamics import forward_dynamics_core
from jbdl.rbdl.contact import detect_contact_core
from jbdl.rbdl.dynamics.state_fun_ode import dynamics_fun_extend_core, events_fun_extend_core
from jbdl.rbdl.dynamics import composite_rigid_body_algorithm_core
from jbdl.rbdl.contact.impulsive_dynamics import impulsive_dynamics_extend_core
from jbdl.rbdl.ode.solve_ivp import integrate_dynamics
from jax.custom_derivatives import closure_convert
import math
from jax.api import jit
from functools import partial
from jbdl.rbdl.tools import plot_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
SCRIPTS_PATH = os.path.dirname(CURRENT_PATH)
MODEL_DATA_PATH = os.path.join(SCRIPTS_PATH, "model_data") 
mdlw = ModelWrapper()
mdlw.load(os.path.join(MODEL_DATA_PATH, 'half_max_v1.json'))
model = mdlw.model



nc = int(model["nc"])
nb = int(model["nb"])
nf = int(model["nf"])
contact_cond = model["contact_cond"]
x_tree = device_put(model["x_tree"])
ST = model["ST"]
contactpoint = model["contactpoint"],
idcontact = tuple(model["idcontact"])
parent = tuple(model["parent"])
jtype = tuple(model["jtype"])
jaxis = xyz2int(model["jaxis"])
contactpoint = model["contactpoint"]
I = device_put(model["I"])
a_grav = device_put(model["a_grav"])
mu = device_put(0.9)
contact_force_lb = device_put(contact_cond["contact_force_lb"])
contact_force_ub = device_put(contact_cond["contact_force_ub"])
contact_pos_lb = contact_cond["contact_pos_lb"]
contact_vel_lb = contact_cond["contact_vel_lb"]
contact_vel_ub = contact_cond["contact_vel_ub"]

q0 = jnp.array([0.0,  0.4125, 0.0, math.pi/6, math.pi/6, -math.pi/3, -math.pi/3])
qdot0 = jnp.zeros((7, ))

q_star = jnp.array([0.0,  0.0, 0.0, math.pi/3, math.pi/3, -2*math.pi/3, -2*math.pi/3])
qdot_star = jnp.zeros((7, ))

x0 = jnp.hstack([q0, qdot0])
t_span = (0.0, 2e-3)
delta_t = 5e-4
tau = 0.0

# flag_contact = (0, 0)
ncp = 0

def dynamics_fun(x, t, x_tree, I, contactpoint, u, a_grav, \
    contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu,\
    ST, idcontact,   parent, jtype, jaxis, nb, nc, nf, ncp):
    q = x[0:nb]
    qdot = x[nb:]
    tau = jnp.matmul(ST, u)
    flag_contact = detect_contact_core(x_tree, q, qdot, contactpoint, contact_pos_lb, contact_vel_lb, contact_vel_ub,\
        idcontact, parent, jtype, jaxis, nc)
    xdot,fqp, H = dynamics_fun_extend_core(x_tree, I, q, qdot, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub,\
    idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, ncp, mu)
    return xdot

def events_fun(y, t, x_tree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub, \
    contact_pos_lb, contact_vel_lb, contact_vel_ub, mu, ST, idcontact,  parent, jtype, jaxis, nb, nc, nf, ncp):
    q = y[0:nb]
    qdot = y[nb:]
    flag_contact = detect_contact_core(x_tree, q, qdot, contactpoint, contact_pos_lb, contact_vel_lb, contact_vel_ub,\
        idcontact, parent, jtype, jaxis, nc)

    value = events_fun_extend_core(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nc)
    return value

def impulsive_dynamics_fun(y, t, x_tree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub, \
    contact_pos_lb, contact_vel_lb, contact_vel_ub, mu, ST, idcontact,  parent, jtype, jaxis, nb, nc, nf, ncp):
    q = y[0:nb]
    qdot = y[nb:]
    H =  composite_rigid_body_algorithm_core(x_tree, I, parent, jtype, jaxis, nb, q)
    flag_contact = detect_contact_core(x_tree, q, qdot, contactpoint, contact_pos_lb, contact_vel_lb, contact_vel_ub,\
        idcontact, parent, jtype, jaxis, nc)
    qdot_impulse = impulsive_dynamics_extend_core(x_tree, q, qdot, contactpoint, H, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf)
    qdot_impulse = qdot_impulse.flatten()
    y_new = jnp.hstack([q, qdot_impulse])
    return y_new

pure_dynamics_fun = partial(dynamics_fun, ST=ST, idcontact=idcontact, \
        parent=parent, jtype=jtype, jaxis=jaxis, nb=nb, nc=nc, nf=nf, ncp=ncp)

pure_events_fun = partial(events_fun, ST=ST, idcontact=idcontact, \
        parent=parent, jtype=jtype, jaxis=jaxis, nb=nb, nc=nc, nf=nf, ncp=ncp)

pure_impulsive_fun =  partial(impulsive_dynamics_fun, ST=ST, idcontact=idcontact, \
    parent=parent, jtype=jtype, jaxis=jaxis, nb=nb, nc=nc, nf=nf, ncp=ncp)


def dynamics_step(pure_dynamics_fun, y0, t_span, delta_t, event, impulsive, *args):
    t_eval, sol =  integrate_dynamics(pure_dynamics_fun, y0, t_span, delta_t, event, impulsive, args=args)
    yT = sol[-1, :]
    return yT


u = jnp.zeros((4,))
pure_args = (x_tree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu)

print(dynamics_step(pure_dynamics_fun, x0, t_span, delta_t, pure_events_fun, pure_impulsive_fun, *pure_args))
# %%
xk = jnp.array([0.00652783, 0.2478803, -0.06356288, \
    1.0361857, 1.0312966, -2.095648, -2.098419, \
    0.00470335, -1.7301282,  -0.03855195, \
    0.3507634, 0.19818218, 0.05037037, 0.05680534])
u = jnp.array([0.19982854, 0.5968663, 0.01227411, 0.14438418])
xkp1 = dynamics_step(pure_dynamics_fun, xk, t_span, delta_t, pure_events_fun, pure_impulsive_fun, *pure_args)
# print(pure_impulsive_fun(xk, 0.1, *pure_args))
# q = xk[nb:]
# H =  composite_rigid_body_algorithm_core(x_tree, I, parent, jtype, jaxis, nb, q)
# print(H)

# %%
%matplotlib 

q0 = np.array([0,  0.5, 0, math.pi/6, -math.pi/6, -math.pi/3, math.pi/3]) # stand with leg in
qdot0 = jnp.zeros((7, ))
x0 = jnp.hstack([q0, qdot0])

q_star = jnp.array([0.0,  0.0, 0.0, math.pi/6, -math.pi/6, -math.pi/3, math.pi/3])
qdot_star = jnp.zeros((7, ))

kp = 200
kd = 3
kp = 50
kd = 1
xksv = []
T = 2e-3

xk = x0
plt.figure()
plt.ion()

fig = plt.gcf()
ax = Axes3D(fig)


for i in range(500):
    print(i)
    u = kp * (q_star[3:7] - xk[3:7]) + kd * (qdot_star[3:7] - xk[10:14])
    pure_args = (x_tree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub, contact_pos_lb, contact_vel_lb, contact_vel_ub,mu)
    # print("xk:", xk)
    # print("u", u)
    xk = dynamics_step(pure_dynamics_fun, xk, t_span, delta_t, pure_events_fun, pure_impulsive_fun, *pure_args)


    # xksv.append(xk)
    ax.clear()
    plot_model(model, xk[0:7], ax)
    # fcqp = np.array([0, 0, 1, 0, 0, 1])
    # plot_contact_force(model, xk[0:7], contact_force["fc"], contact_force["fcqp"], contact_force["fcpd"], 'fcqp', ax)
    ax.view_init(elev=0,azim=-90)
    ax.set_xlabel('X')
    ax.set_xlim(-0.3, -0.3+0.6)
    ax.set_ylabel('Y')
    ax.set_ylim(-0.15, -0.15+0.6)
    ax.set_zlabel('Z')
    ax.set_zlim(-0.1, -0.1+0.6)
    ax.set_title('Frame')
    plt.pause(1e-8)
    # fig.canvas.draw()
plt.ioff()
# %%
