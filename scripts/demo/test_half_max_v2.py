import os

import jax
from jbdl.rbdl.utils import ModelWrapper
from jax import device_put
from jbdl.rbdl.utils import xyz2int
import jax.numpy as jnp
from jbdl.rbdl.dynamics import forward_dynamics_core
from jbdl.rbdl.dynamics.state_fun_ode import dynamics_fun_core
from jbdl.rbdl.ode.solve_ivp import integrate_dynamics
from jax.custom_derivatives import closure_convert
import math
from jax.api import jit
from functools import partial
from jbdl.rbdl.tools import plot_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
SCRIPTS_PATH = os.path.dirname(CURRENT_PATH)
MODEL_DATA_PATH = os.path.join(SCRIPTS_PATH, "model_data") 
mdlw = ModelWrapper()
mdlw.load(os.path.join(MODEL_DATA_PATH, 'half_max_v1.json'))
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

q0 = jnp.array([0.0,  0.4125, 0.0, math.pi/6, math.pi/6, -math.pi/3, -math.pi/3])
qdot0 = jnp.zeros((7, ))

q_star = jnp.array([0.0,  0.0, 0.0, math.pi/3, math.pi/3, -2*math.pi/3, -2*math.pi/3])
qdot_star = jnp.zeros((7, ))

x0 = jnp.hstack([q0, qdot0])
t_span = (0.0, 2e-3)
delta_t = 5e-4
tau = 0.0

flag_contact = (0, 0, 0, 0)
rankJc = 0
ncp = 0

def dynamics_fun(x, t, Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub, mu,\
    ST, idcontact, flag_contact,  parent, jtype, jaxis, NB, NC, nf, rankJc, ncp):
    q = x[0:NB]
    qdot = x[NB:]
    tau = jnp.matmul(ST, u)
    xdot,fqp, H = dynamics_fun_core(Xtree, I, q, qdot, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub,\
    idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, rankJc, ncp, mu)
    return xdot

pure_dynamics_fun = partial(dynamics_fun, ST=ST, idcontact=idcontact, flag_contact=flag_contact, \
    parent=parent, jtype=jtype, jaxis=jaxis, NB=NB, NC=NC, nf=nf, rankJc=rankJc, ncp=ncp)

def dynamics_step(y0, t_span, delta_t, event, *args):
    t_eval, sol =  integrate_dynamics(pure_dynamics_fun, y0, t_span, delta_t, event, args=args)
    yT = sol[-1, :]
    return yT

u = jnp.zeros((4,))
pure_args = (Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub, mu)

print(dynamics_step(x0, t_span, delta_t, None, *pure_args))

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


for i in range(200):
    print(i)
    u = kp * (q_star[3:7] - xk[3:7]) + kd * (qdot_star[3:7] - xk[10:14])
    pure_args = (Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub, mu)
    xk = dynamics_step(xk, t_span, delta_t, None, *pure_args)


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

# args = (x0, t0, Xtree, I,  a_grav)

# def forward_dynamics(x, t, Xtree, I, a_grav, parent, jtype, jaxis, NB):
#     q = jnp.zeros((7,))
#     qdot = jnp.zeros((7,))
#     ttau = jnp.zeros((7,))
#     qddot = forward_dynamics_core(Xtree, I, parent, jtype, jaxis, NB, q, qdot, ttau, a_grav)
#     return qddot

# test = partial(forward_dynamics, parent0=parent, jtype0=jtype, jaxis0=jaxis, NB0=NB)
# test(*args)
# converted, consts = closure_convert(partial(forward_dynamics, parent=parent, jtype=jtype, jaxis=jaxis, NB=NB), *args)
