# %%
# Load model.
import os

from jax.linear_util import transformation_with_aux
from jbdl.rbdl.utils import ModelWrapper

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
SCRIPTS_PATH = os.path.dirname(CURRENT_PATH)
MODEL_DATA_PATH = os.path.join(SCRIPTS_PATH, "model_data") 
mdlw = ModelWrapper()
mdlw.load(os.path.join(MODEL_DATA_PATH, 'half_max_v1.json'))
model = mdlw.model

# %%
# Plot initial state.
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from jbdl.rbdl.tools import plot_model
%matplotlib inline
q = np.array([0.0,  0.4125, 0.0, math.pi/6, math.pi/6, -math.pi/3, -math.pi/3])
fig = plt.figure()
ax = Axes3D(fig)
ax.set_ylim3d(-1.0, 1.0)
plot_model(model, q, ax)
ax.view_init(elev=0,azim=-90)
ax.set_xlabel('X')
ax.set_xlim(-0.3, -0.3+0.8)
ax.set_ylabel('Y')
ax.set_ylim(-0.15, -0.15+0.8)
ax.set_zlabel('Z')
ax.set_zlim(-0.1, -0.1+0.8)
plt.show()

# %%
# Load model parameters.
from jax import device_put
from jbdl.rbdl.utils import xyz2int
NC = int(model["NC"])
NB = int(model["NB"])
nf = int(model["nf"])
contact_cond = model["contact_cond"]
Xtree = device_put(model["Xtree"])
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
contact_pos_lb = device_put(contact_cond["contact_pos_lb"])
contact_vel_lb = device_put(contact_cond["contact_vel_lb"])
contact_vel_ub = device_put(contact_cond["contact_vel_ub"])
ST = device_put(model["ST"])

# %%
from jbdl.rbdl.dynamics.state_fun_ode import dynamics_fun_extend_core, events_fun_extend_core
import jax.numpy as jnp
from functools import partial
from jax.custom_derivatives import closure_convert
from jbdl.rbdl.contact import detect_contact_core
from jbdl.rbdl.dynamics import composite_rigid_body_algorithm_core
from jbdl.rbdl.contact.impulsive_dynamics import impulsive_dynamics_extend_core

q_star = jnp.array([0.0,  0.0, 0.0, math.pi/6, math.pi/6, -math.pi/3, -math.pi/3])
qdot_star = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0])
# tau = jnp.zeros((7,))
u = jnp.zeros((4,))

# flag_contact = jnp.array([1, 1])
rankJc = 0
ncp = 0

q = jnp.array([0.0, 0.4125, 0.0, math.pi/6, math.pi/6, -math.pi/3, -math.pi/3])
qdot = jnp.ones((7, ))
x = jnp.hstack([q, qdot])
# xdot, fqp, H = dynamics_fun_core(Xtree, I, q, qdot, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub,\
#     idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, rankJc, ncp, mu)

# def dynamics_fun(x, t, Xtree, I, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub, mu,\
#     flag_contact, idcontact,  parent, jtype, jaxis, NB, NC, nf, ncp):
#     q = x[0:NB]
#     qdot = x[NB:]
#     xdot,fqp, H = dynamics_fun_extend_core(Xtree, I, q, qdot, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub,\
#     idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, ncp, mu)
#     return xdot

def dynamics_fun(x, t, Xtree, I, contactpoint, u, a_grav, \
    contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu,\
    ST, idcontact,   parent, jtype, jaxis, NB, NC, nf, ncp):
    q = x[0:NB]
    qdot = x[NB:]
    tau = jnp.matmul(ST, u)
    flag_contact = detect_contact_core(Xtree, q, qdot, contactpoint, contact_pos_lb, contact_vel_lb, contact_vel_ub,\
        idcontact, parent, jtype, jaxis, NC)
    xdot,fqp, H = dynamics_fun_extend_core(Xtree, I, q, qdot, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub,\
    idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, ncp, mu)
    return xdot

def events_fun(y, t, Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub, \
    contact_pos_lb, contact_vel_lb, contact_vel_ub, mu, ST, idcontact,  parent, jtype, jaxis, NB, NC, nf, ncp):
    q = y[0:NB]
    qdot = y[NB:]
    flag_contact = detect_contact_core(Xtree, q, qdot, contactpoint, contact_pos_lb, contact_vel_lb, contact_vel_ub,\
        idcontact, parent, jtype, jaxis, NC)

    value = events_fun_extend_core(Xtree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NC)
    return value

def impulsive_dynamics_fun(y, t, Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub, \
    contact_pos_lb, contact_vel_lb, contact_vel_ub, mu, ST, idcontact,  parent, jtype, jaxis, NB, NC, nf, ncp):
    q = y[0:NB]
    qdot = y[NB:]
    H =  composite_rigid_body_algorithm_core(Xtree, I, parent, jtype, jaxis, NB, q)
    flag_contact = detect_contact_core(Xtree, q, qdot, contactpoint, contact_pos_lb, contact_vel_lb, contact_vel_ub,\
        idcontact, parent, jtype, jaxis, NC)
    qdot_impulse = impulsive_dynamics_extend_core(Xtree, q, qdot, contactpoint, H, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
    qdot_impulse = qdot_impulse.flatten()
    y_new = jnp.hstack([q, qdot_impulse])
    return y_new


t = device_put(0.0)
xdot = dynamics_fun(x, t, Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub, \
    contact_pos_lb, contact_vel_lb, contact_vel_ub, mu, ST, idcontact,  parent, jtype, jaxis, NB, NC, nf, ncp)

# print(xdot)


pure_dynamics_fun = partial(dynamics_fun, ST=ST, idcontact=idcontact, \
        parent=parent, jtype=jtype, jaxis=jaxis, NB=NB, NC=NC, nf=nf, ncp=ncp)

pure_events_fun = partial(events_fun, ST=ST, idcontact=idcontact, \
        parent=parent, jtype=jtype, jaxis=jaxis, NB=NB, NC=NC, nf=nf, ncp=ncp)

pure_impulsive_fun =  partial(impulsive_dynamics_fun, ST=ST, idcontact=idcontact, \
        parent=parent, jtype=jtype, jaxis=jaxis, NB=NB, NC=NC, nf=nf, ncp=ncp)

pure_args = (x, t, Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu)

# pure_args = (x, t, Xtree, I, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub, mu, flag_contact)

converted, consts = closure_convert(pure_dynamics_fun, *pure_args)
converted, consts = closure_convert(pure_events_fun, *pure_args)
converted, consts = closure_convert(pure_impulsive_fun, *pure_args)

print(pure_dynamics_fun(*pure_args))
print(pure_events_fun(*pure_args))
print(pure_impulsive_fun(*pure_args))




# %%
%%time
from jbdl.rbdl.ode.solve_ivp import integrate_dynamics
from jax.api import jacrev
import time
q0 = jnp.array([0.0,  0.4125, 0.0, math.pi/6, math.pi/6, -math.pi/3, -math.pi/3])
qdot0 = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0])
x0 = jnp.hstack([q0, qdot0])
t_span = (0.0, 2e-3)
delta_t = 5e-4

pure_args = (Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu)
# pure_args = (Xtree, I, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub, mu, flag_contact)

t_eval, sol = integrate_dynamics(pure_dynamics_fun, x0, t_span, delta_t, args=pure_args)

def dynamics_step(y0, t_span, delta_t, event, impulsive, *args):
    t_eval, sol =  integrate_dynamics(pure_dynamics_fun, y0, t_span, delta_t, event, impulsive, args=args)
    yT = sol[-1, :]
    return yT


print(dynamics_step(x0, t_span, delta_t, pure_events_fun, pure_impulsive_fun, *pure_args))
# start_time = time.time()
# dxTdx0_func = jacrev(dynamics_step, argnums=[0,])
# dxTdx0 = dxTdx0_func(x0, t_span, delta_t, None, *pure_args)
# dxTdx0.block_until_ready()
# print(dxTdx0)
# duration = time.time() - start_time
# print(duration)

# %%
# start_time = time.time()
# dxTdx0 = dxTdx0_func(x0, t_span, delta_t, None, *pure_args)
# print(dxTdx0)
# duration = time.time() - start_time
# print(duration)



# %%
# from jax.custom_derivatives import closure_convert
# from jax.api_util import flatten_fun_nokwargs
# from jax.interpreters import partial_eval as pe
# from jax import linear_util as lu
# from jbdl.rbdl.dynamics import forward_dynamics_core
# from functools import partial
# t0 = device_put(0.0)

# args = (x0, t0, Xtree, I, a_grav)


# def forward_dynamics(x, t, Xtree, I,  a_grav, parent, jtype, jaxis, NB):
#     q = x[0:NB]
#     qdot = x[NB:]
#     ttau = jnp.zeros((7,))
#     qddot = forward_dynamics_core(Xtree, I, parent, jtype, jaxis, NB, q, qdot, ttau, a_grav)
#     return qddot

# pure_forward_dynamics = partial(forward_dynamics, parent=parent, jtype=jtype, jaxis=jaxis, NB=NB)

# converted, consts = closure_convert(pure_forward_dynamics, *args)

# %%


