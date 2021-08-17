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
from jbdl.rbdl.contact import detect_contact, detect_contact_core
from jbdl.rbdl.contact import impulsive_dynamics, impulsive_dynamics_core
from jbdl.rbdl.dynamics import composite_rigid_body_algorithm_core, forward_dynamics_core, inverse_dynamics_core
from jbdl.rbdl.kinematics import *
from jbdl.rbdl.kinematics import calc_body_to_base_coordinates_core
import time
from jbdl.rbdl.utils import xyz2int
from jax.api import device_put
import jax.numpy as jnp
from functools import partial
# matplotlib.use('TkAgg')

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
SCRIPTS_PATH = os.path.dirname(CURRENT_PATH)
MODEL_DATA_PATH = os.path.join(SCRIPTS_PATH, "model_data") 

mdlw = ModelWrapper()
mdlw.load(os.path.join(MODEL_DATA_PATH, 'whole_max_v1.json'))
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
tau = 0.0

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


u = jnp.zeros((12,))
pure_args = (x_tree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu)

print(dynamics_step(pure_dynamics_fun, x0, t_span, delta_t, pure_events_fun, pure_impulsive_fun, *pure_args))



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
    pure_args = (x_tree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub, contact_pos_lb, contact_vel_lb, contact_vel_ub,mu)
    # print("xk:", xk)
    # print("u", u)
    xk = dynamics_step(pure_dynamics_fun, xk, t_span, delta_t, pure_events_fun, pure_impulsive_fun, *pure_args)


    # xksv.append(xk)
    ax.clear()
    plot_model(model, xk[0:18], ax)
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
