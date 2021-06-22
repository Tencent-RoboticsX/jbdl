# %%
import os
import re
from jax.interpreters.xla import jaxpr_replicas
from numpy.core.shape_base import block
import numpy as np
import math
from jbdl.rbdl.kinematics import calc_pos_vel_point_to_base
from jbdl.rbdl.kinematics import calc_whole_body_com
from jbdl.rbdl.tools import plot_model, plot_contact_force, plot_com_inertia
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from jbdl.rbdl.dynamics.state_fun_ode import events_fun_extend_core, dynamics_fun_extend_core
from jbdl.rbdl.contact.impulsive_dynamics import impulsive_dynamics_extend_core
from jbdl.rbdl.ode.solve_ivp import integrate_dynamics
import matplotlib
from jbdl.rbdl.utils import ModelWrapper
from jbdl.rbdl.contact import detect_contact, detect_contact_core, get_contact_fcqp
from jbdl.rbdl.contact import impulsive_dynamics, impulsive_dynamics_core
from jbdl.rbdl.dynamics import composite_rigid_body_algorithm_core, forward_dynamics_core, inverse_dynamics_core
from jbdl.rbdl.kinematics import *
from jbdl.rbdl.kinematics import calc_body_to_base_coordinates_core
import time
from jbdl.rbdl.utils import xyz2int
from jax.api import device_put
import jax.numpy as jnp
from functools import partial
from jbdl.experimental.ode.solve_ivp import solve_ivp
from jax import jit
# matplotlib.use('TkAgg')

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
SCRIPTS_PATH = os.path.dirname(CURRENT_PATH)
MODEL_DATA_PATH = os.path.join(SCRIPTS_PATH, "model_data") 

mdlw = ModelWrapper()
mdlw.load(os.path.join(MODEL_DATA_PATH, 'whole_max_v1.json'))
model = mdlw.model

NC = int(model["NC"])
NB = int(model["NB"])
nf = int(model["nf"])
contact_cond = model["contact_cond"]
Xtree = device_put(model["Xtree"])
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

q0 = jnp.array([0, 0, 0.5, 0.0, 0, 0,
        0, 0.5, -0.8,  # fr
        0, 0.5, -0.8,  # fl
        0, 0.5, -0.8,  # br
        0, 0.5, -0.8]) # bl
qdot0 = jnp.zeros((18, ))

q_star = jnp.array([0, 0, 0.5, 0.0, 0, 0,
        0, 0.5, -0.8,  # fr
        0, 0.5, -0.8,  # fl
        0, 0.5, -0.8,  # br
        0, 0.5, -0.8]) # bl
qdot_star = jnp.zeros((18, ))

x0 = jnp.hstack([q0, qdot0])
t_span = (0.0, 2e-3)
delta_t = 5e-4
t_eval = jnp.linspace(0.0, 2e-3, 4)
tau = 0.0

ncp = 0

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

def fqp_fun(x, t, Xtree, I, contactpoint, u, a_grav, \
    contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu,\
    ST, idcontact,   parent, jtype, jaxis, NB, NC, nf, ncp):
    q = x[0:NB]
    qdot = x[NB:]
    tau = jnp.matmul(ST, u)
    flag_contact = detect_contact_core(Xtree, q, qdot, contactpoint, contact_pos_lb, contact_vel_lb, contact_vel_ub,\
        idcontact, parent, jtype, jaxis, NC)
    xdot, fqp, H = dynamics_fun_extend_core(Xtree, I, q, qdot, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub,\
    idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, ncp, mu)
    return fqp, flag_contact

pure_dynamics_fun = partial(dynamics_fun, ST=ST, idcontact=idcontact, \
        parent=parent, jtype=jtype, jaxis=jaxis, NB=NB, NC=NC, nf=nf, ncp=ncp)

pure_events_fun = partial(events_fun, ST=ST, idcontact=idcontact, \
        parent=parent, jtype=jtype, jaxis=jaxis, NB=NB, NC=NC, nf=nf, ncp=ncp)

pure_impulsive_fun =  partial(impulsive_dynamics_fun, ST=ST, idcontact=idcontact, \
    parent=parent, jtype=jtype, jaxis=jaxis, NB=NB, NC=NC, nf=nf, ncp=ncp)

pure_fqp_fun = partial(fqp_fun, ST=ST, idcontact=idcontact, \
    parent=parent, jtype=jtype, jaxis=jaxis, NB=NB, NC=NC, nf=nf, ncp=ncp)



u = jnp.zeros((12,))
pure_args = (Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu)

print(solve_ivp(pure_dynamics_fun, x0, t_eval, pure_events_fun, pure_impulsive_fun, *pure_args))



# %%
%matplotlib 

q0 = jnp.array([0, 0, 0.5, 0.0, 0, 0,
        0, 0.5, -0.8,  # fr
        0, 0.5, -0.8,  # fl
        0, 0.5, -0.8,  # br
        0, 0.5, -0.8]) # bl
qdot0 = jnp.zeros((18, ))
x0 = jnp.hstack([q0, qdot0])

q_star = jnp.array([0, 0, 0.5, 0.0, 0, 0,
        0, 0.5, -0.8,  # fr
        0, 0.5, -0.8,  # fl
        0, 0.5, -0.8,  # br
        0, 0.5, -0.8]) # bl
qdot_star = jnp.zeros((18, ))


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
    # u = kp * (q_star[3:7] - xk[3:7]) + kd * (qdot_star[3:7] - xk[10:14])
    u = kp * (q_star[6:18] - xk[6:18]) + kd * (qdot_star[6:18] - xk[24:36])
    pure_args = (Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub, contact_pos_lb, contact_vel_lb, contact_vel_ub,mu)
    # print("xk:", xk)
    # print("u", u)
    xk = solve_ivp(pure_dynamics_fun, xk, t_eval, pure_events_fun, pure_impulsive_fun, *pure_args)[-1,:]

    fqp, flag_contact = jit(pure_fqp_fun)(xk, 0.0, *pure_args)
    fcqp = get_contact_fcqp(fqp, flag_contact, NC, nf)
    # print(fcqp)

    # xksv.append(xk)
    ax.clear()
    plot_model(model, xk[0:18], ax)
    # fcqp = np.array([0, 0, 1, 0, 0, 1])
    plot_contact_force(model, xk[0:18], None, fcqp, None, 'fcqp', ax)
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
