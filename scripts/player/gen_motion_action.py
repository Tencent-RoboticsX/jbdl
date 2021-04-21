# %%
from io import IncrementalNewlineDecoder
import matplotlib
import numpy as np
import os
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
SCRIPTS_PATH = os.path.dirname(CURRENT_PATH)
MOTION_DATA_PATH = os.path.join(SCRIPTS_PATH, "motion_data")
MODEL_DATA_PATH = os.path.join(SCRIPTS_PATH, "model_data")

# %%
# Load motion data.
import math
x = np.loadtxt(os.path.join(MOTION_DATA_PATH, "half_max_hard_motion_data.dat"))
q = x[0:7, :]
qdot = x[7:14, :]
x0 = x[:, 0:1]
q0 = q[:, 0:1]
qdot0 = qdot[:, 0:1]
q_star = np.array([0.0,  0.4125, 0.0, math.pi/4, math.pi/4, -2*math.pi/4, -2*math.pi/4])
q_star = q_star.reshape(-1, 1)
qdot_star = np.zeros((7, 1))
u0 = np.zeros((4, 1))
# %%
from jbdl.rbdl.utils import ModelWrapper
mdlw = ModelWrapper()
mdlw.load(os.path.join(MODEL_DATA_PATH, 'half_max_v1.json'))
model = mdlw.model
# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from jbdl.rbdl.tools import plot_model
%matplotlib inline
fig = plt.figure()
ax = Axes3D(fig)
ax.set_ylim3d(-1.0, 1.0)
plot_model(model, q_star, ax)
ax.view_init(elev=0,azim=-90)
ax.set_xlabel('X')
ax.set_xlim(-0.3, -0.3+0.8)
ax.set_ylabel('Y')
ax.set_ylim(-0.15, -0.15+0.8)
ax.set_zlabel('Z')
ax.set_zlim(-0.1, -0.1+0.8)
plt.show()

# %%
from jbdl.rbdl.dynamics.state_fun_ode import state_fun_ode
from jbdl.rbdl.tools import plot_contact_force
%matplotlib
fig = plt.figure()
ax = Axes3D(fig)
xk = np.vstack([q0, qdot0])
kp = 200
kd = 3
kp = 50
kd = 1
xksv = []
T = 2e-3
for i in range(1000):
    print(i)
    u = kp * (q_star[3:7] - xk[3:7]) + kd * (qdot_star[3:7] - xk[10:14])
    xk, contact_force = state_fun_ode(model, xk.flatten(), u.flatten(), T)
    xk = xk.reshape(-1, 1)

    xksv.append(xk)
    ax.clear()
    plot_model(model, xk[0:7], ax)
    # fcqp = np.array([0, 0, 1, 0, 0, 1])
    plot_contact_force(model, xk[0:7], contact_force["fc"], contact_force["fcqp"], contact_force["fcpd"], 'fcqp', ax)
    ax.view_init(elev=0,azim=-90)
    ax.set_xlabel('X')
    ax.set_xlim(-0.3, -0.3+0.6)
    ax.set_ylabel('Y')
    ax.set_ylim(-0.15, -0.15+0.6)
    ax.set_zlabel('Z')
    ax.set_zlim(-0.1, -0.1+0.6)
    ax.set_title('Frame')
    plt.pause(1e-8)
    plt.show() 