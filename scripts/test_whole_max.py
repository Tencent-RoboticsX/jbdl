import os
import re
from numpy.core.shape_base import block
import numpy as np
import math
from jaxRBDL.Kinematics.CalcPosVelPointToBase import CalcPosVelPointToBase
from jaxRBDL.Kinematics.CalcWholeBodyCoM import CalcWholeBodyCoM
from jaxRBDL.Tools.PlotModel import PlotModel
from jaxRBDL.Tools.PlotContactForce import PlotContactForce
from jaxRBDL.Tools.PlotCoMInertia import PlotCoMInertia
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from jaxRBDL.Dynamics.StateFunODE import StateFunODE
import matplotlib
from jaxRBDL.Utils.ModelWrapper import ModelWrapper
matplotlib.use('TkAgg')

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
print(CURRENT_PATH)
mdlw = ModelWrapper()
mdlw.load(os.path.join(CURRENT_PATH, 'whole_max_v1.json'))
print(type(mdlw))
model = mdlw.model

q = np.array([0, 0, 0.27, 0, 0, 0,
    0, 0.5, -0.8,  # fr
    0, 0.5, -0.8,  # fl
    0, 0.5, -0.8,  # br
    0, 0.5, -0.8]) # bl
# q = np.array([0.0,  0.4125, 0.0, math.pi/6, math.pi/6, -math.pi/3, -math.pi/3]) # stand high
# q = np.array([0.0,  0.2382, 0.0, math.pi/3, math.pi/3, -2*math.pi/3, -2*math.pi/3])  # stand low
# q = np.array([0.0,  0.456, -math.pi/2, math.pi/6, 2.7487, 0, -2.0070]) # stand end

qdot = np.zeros((18, 1))

# Pcom = CalcWholeBodyCoM(model, q)
# print(Pcom)

plt.figure()
plt.ion()

fig = plt.gcf()
ax = Axes3D(fig)
ax.set_ylim3d(-1.0, 1.0)
PlotModel(model, q, ax)
PlotCoMInertia(model, q, ax)
ax.view_init(elev=0,azim=-90)
ax.set_xlabel('X')
ax.set_xlim(-0.3, -0.3+0.8)
ax.set_ylabel('Y')
ax.set_ylim(-0.15, -0.15+0.8)
ax.set_zlabel('Z')
ax.set_zlim(-0.1, -0.1+0.8)
# plt.show()

idcontact = model["idcontact"]
contactpoint = model["contactpoint"]
idbase = 6

# pos, vel = CalcPosVelPointToBase(model, q, qdot, idcontact[0], idbase, contactpoint[0])
# pos, vel = CalcPosVelPointToBase(model, q, qdot, idcontact[1], idbase, contactpoint[1])

# q0 = np.array([0,  0.4127, 0, math.pi/6, math.pi/6, -math.pi/3, -math.pi/3]) # stand high
# q0 = np.array([0,  0.45, 0, -math.pi/6, math.pi/6, math.pi/3, -math.pi/3]) # stand with leg out
# q0 = np.array([0,  0.4127, 0, math.pi/6, -math.pi/6, -math.pi/3, math.pi/3]) # stand with leg in
# q0 = np.array([0,  0.43, 0, math.pi/6, -math.pi/6, -math.pi/3, math.pi/3]) # stand with leg in
# q0 = np.array([0,  0.2382, 0, math.pi/3, math.pi/3, -2*math.pi/3, -2*math.pi/3]) # stand low
q0 = np.array([0, 0, 0.5, 0.0, 0, 0,
    0, 0.5, -0.8,  # fr
    0, 0.5, -0.8,  # fl
    0, 0.5, -0.8,  # br
    0, 0.5, -0.8]) # bl

q0 = q0.reshape(-1, 1)


ax.clear()
PlotModel(model, q0, ax)
ax.view_init(elev=0,azim=-90)
ax.set_xlabel('X')
ax.set_xlim(-0.3, -0.3+0.6)
ax.set_ylabel('Y')
ax.set_ylim(-0.15, -0.15+0.6)
ax.set_zlabel('Z')
ax.set_zlim(-0.1, -0.1+0.6)
fig.show()
# plt.show()


qd0 = np.zeros((18, 1))
x0 = np.vstack([q0, qd0])
u0 = np.zeros((12, 1))

xk = x0
u = u0
kp = 200
kd = 3
kp = 50
kd = 1
xksv = []
T = 2e-3
xksv = []



for i in range(100):
    print(i)
    u = kp * (q0[6:18] - xk[6:18]) + kd * (qd0[6:18] - xk[24:36])
    xk, contact_force = StateFunODE(model, xk.flatten(), u.flatten(), T)
    xk = xk.reshape(-1, 1)

    xksv.append(xk)
    ax.clear()
    PlotModel(model, xk[0:18], ax)
    # fcqp = np.array([0, 0, 1, 0, 0, 1])
    # print(contact_force["fcqp"])
    PlotContactForce(model, xk[0:18], contact_force["fc"], contact_force["fcqp"], contact_force["fcpd"], 'fcqp', ax)
    ax.view_init(elev=0,azim=-90)
    ax.set_xlabel('X')
    ax.set_xlim(-0.3, -0.3+0.6)
    ax.set_ylabel('Y')
    ax.set_ylim(-0.15, -0.15+0.6)
    ax.set_zlabel('Z')
    ax.set_zlim(-0.1, -0.1+0.6)
    ax.set_title('Frame')
    plt.pause(1e-8)
    fig.canvas.draw() 
    # plt.show()


plt.ioff()