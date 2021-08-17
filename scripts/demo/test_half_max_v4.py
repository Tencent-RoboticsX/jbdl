# %%
# Load model.
import os
from jax._src.numpy.lax_numpy import diff
from jax.api import jacfwd

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
nc = int(model["nc"])
nb = int(model["nb"])
nf = int(model["nf"])
contact_cond = model["contact_cond"]
x_tree = device_put(model["x_tree"])
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
from jbdl.rbdl.contact import get_contact_fcqp
import jax

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
# xdot, fqp, H = dynamics_fun_core(x_tree, I, q, qdot, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub,\
#     idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, rankJc, ncp, mu)

# def dynamics_fun(x, t, x_tree, I, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub, mu,\
#     flag_contact, idcontact,  parent, jtype, jaxis, nb, nc, nf, ncp):
#     q = x[0:nb]
#     qdot = x[nb:]
#     xdot,fqp, H = dynamics_fun_extend_core(x_tree, I, q, qdot, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub,\
#     idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, ncp, mu)
#     return xdot

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

def fqp_fun(x, t, x_tree, I, contactpoint, u, a_grav, \
    contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu,\
    ST, idcontact,   parent, jtype, jaxis, nb, nc, nf, ncp):
    q = x[0:nb]
    qdot = x[nb:]
    tau = jnp.matmul(ST, u)
    flag_contact = detect_contact_core(x_tree, q, qdot, contactpoint, contact_pos_lb, contact_vel_lb, contact_vel_ub,\
        idcontact, parent, jtype, jaxis, nc)
    xdot, fqp, H = dynamics_fun_extend_core(x_tree, I, q, qdot, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub,\
    idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, ncp, mu)
    return fqp, flag_contact

t = device_put(0.0)
xdot = dynamics_fun(x, t, x_tree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub, \
    contact_pos_lb, contact_vel_lb, contact_vel_ub, mu, ST, idcontact,  parent, jtype, jaxis, nb, nc, nf, ncp)

fqp = fqp_fun(x, t, x_tree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub, \
    contact_pos_lb, contact_vel_lb, contact_vel_ub, mu, ST, idcontact,  parent, jtype, jaxis, nb, nc, nf, ncp)
print(fqp)


pure_dynamics_fun = partial(dynamics_fun, ST=ST, idcontact=idcontact, \
        parent=parent, jtype=jtype, jaxis=jaxis, nb=nb, nc=nc, nf=nf, ncp=ncp)

pure_events_fun = partial(events_fun, ST=ST, idcontact=idcontact, \
        parent=parent, jtype=jtype, jaxis=jaxis, nb=nb, nc=nc, nf=nf, ncp=ncp)

pure_impulsive_fun =  partial(impulsive_dynamics_fun, ST=ST, idcontact=idcontact, \
        parent=parent, jtype=jtype, jaxis=jaxis, nb=nb, nc=nc, nf=nf, ncp=ncp)

pure_fqp_fun = partial(fqp_fun, ST=ST, idcontact=idcontact, \
        parent=parent, jtype=jtype, jaxis=jaxis, nb=nb, nc=nc, nf=nf, ncp=ncp)

pure_args = (x, t, x_tree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu)

# pure_args = (x, t, x_tree, I, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub, mu, flag_contact)

converted, consts = closure_convert(pure_dynamics_fun, *pure_args)
converted, consts = closure_convert(pure_events_fun, *pure_args)
converted, consts = closure_convert(pure_impulsive_fun, *pure_args)
converted, consts = closure_convert(pure_fqp_fun, *pure_args)

print(pure_dynamics_fun(*pure_args))
print(pure_events_fun(*pure_args))
print(pure_impulsive_fun(*pure_args))
fqp, flag_contact = jax.jit(pure_fqp_fun)(*pure_args)
fcqp = get_contact_fcqp(fqp, flag_contact, nc, nf)
print(fcqp)

#%%
import time
start = time.time()
fqp, flag_contact = jax.jit(pure_fqp_fun)(*pure_args)
fcqp = get_contact_fcqp(fqp, flag_contact, nc, nf)
print(fcqp)
duration = time.time() - start
print("duration:", duration)

start = time.time()
fqp, flag_contact = jax.jit(pure_fqp_fun)(*pure_args)
fcqp = get_contact_fcqp(fqp, flag_contact, nc, nf)
print(fcqp)
duration = time.time() - start
print("duration:", duration)


# %%
from jbdl.experimental.ode.solve_ivp import solve_ivp

q0 = jnp.array([0.0,  0.4125, 0.0, math.pi/6, math.pi/6, -math.pi/3, -math.pi/3])
qdot0 = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0])
x0 = jnp.hstack([q0, qdot0])
t_span = (0.0, 2e-3)
delta_t = 5e-4

pure_args = (x_tree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu)


solve_ivp(pure_dynamics_fun, x0, jnp.linspace(0, 2e-3, 4), pure_events_fun, pure_impulsive_fun, *pure_args)

# %% 

import jax
import time
from jax import jacfwd

print(solve_ivp(pure_dynamics_fun, x0, jnp.linspace(0, 2e-3, 4), pure_events_fun, pure_impulsive_fun, *pure_args))
start = time.time()
diff = jax.jit(jacfwd(solve_ivp, argnums=8), static_argnums=(0, 3, 4))
result = diff(pure_dynamics_fun, x0, jnp.linspace(0, 2e-3, 4), pure_events_fun, pure_impulsive_fun, *pure_args)
result.block_until_ready()
print(result)
duration = time.time() - start
print(duration)

start = time.time()
result = diff(pure_dynamics_fun, x0, jnp.linspace(0, 2e-3, 4), pure_events_fun, pure_impulsive_fun, *pure_args)
result.block_until_ready()
duration = time.time() - start
print(duration)


# %%
import jax
import time
from jax import jacfwd

print(solve_ivp(pure_dynamics_fun, x0, jnp.linspace(0, 2e-3, 4), pure_events_fun, pure_impulsive_fun, *pure_args))
start = time.time()
diff = jax.jit(jacfwd(solve_ivp, argnums=1), static_argnums=(0, 3, 4))
result = diff(pure_dynamics_fun, x0, jnp.linspace(0, 2e-3, 4), pure_events_fun, pure_impulsive_fun, *pure_args)
result.block_until_ready()
duration = time.time() - start
print(duration)

start = time.time()
result = diff(pure_dynamics_fun, x0, jnp.linspace(0, 2e-3, 4), pure_events_fun, pure_impulsive_fun, *pure_args)
result.block_until_ready()
duration = time.time() - start
print(duration)




# %%
# from jax.custom_derivatives import closure_convert
# from jax.api_util import flatten_fun_nokwargs
# from jax.interpreters import partial_eval as pe
# from jax import linear_util as lu
# from jbdl.rbdl.dynamics import forward_dynamics_core
# from functools import partial
# t0 = device_put(0.0)

# args = (x0, t0, x_tree, I, a_grav)


# def forward_dynamics(x, t, x_tree, I,  a_grav, parent, jtype, jaxis, nb):
#     q = x[0:nb]
#     qdot = x[nb:]
#     ttau = jnp.zeros((7,))
#     qddot = forward_dynamics_core(x_tree, I, parent, jtype, jaxis, nb, q, qdot, ttau, a_grav)
#     return qddot

# pure_forward_dynamics = partial(forward_dynamics, parent=parent, jtype=jtype, jaxis=jaxis, nb=nb)

# converted, consts = closure_convert(pure_forward_dynamics, *args)

# %%


