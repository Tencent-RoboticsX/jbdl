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
from jaxRBDL.Dynamics.StateFunODE import DynamicsFunCore, EventsFunCore
from jaxRBDL.Contact.DetectContact import DetectContact, DetectContactCore
from jaxRBDL.Contact.ImpulsiveDynamics import ImpulsiveDynamicsCore
import time
# matplotlib.use('TkAgg')


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

def jit_compiled(model):
    NC = int(model["NC"])
    NB = int(model["NB"])
    nf = int(model["nf"])
    Xtree = model["Xtree"]
    contactpoint = model["contactpoint"],
    idcontact = tuple(model["idcontact"])
    parent = tuple(model["parent"])
    jtype = tuple(model["jtype"])
    jaxis = model["jaxis"]
    contactpoint = model["contactpoint"]
    contact_cond = model["contact_cond"]
    contact_pos_lb = contact_cond["contact_pos_lb"]
    contact_vel_lb = contact_cond["contact_vel_lb"]
    contact_vel_ub = contact_cond["contact_vel_ub"]
    a_grav = model["a_grav"]
    flag_contact_list = [(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)]
    I = model["I"]
    q = np.array([
        0, 0, 0.27, 0, 0, 0, # base
        0, 0.5, -0.8,  # fr
        0, 0.5, -0.8,  # fl
        0, 0.5, -0.8,  # br
        0, 0.5, -0.8]) # bl
    qdot = np.ones(NB)
    # qddot = np.ones(NB)
    tau = np.concatenate([np.zeros(6), np.ones(NB-6)])
    for flag_contact in flag_contact_list:
        print("Jit compiled for %s ..." % str(flag_contact))
        start_time = time.time()
        rankJc = int(np.sum( [1 for item in flag_contact if item != 0]) * model["nf"])
        xdot, fqp, H = DynamicsFunCore(Xtree, I, q, qdot, contactpoint, tau, a_grav, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, rankJc)
        value = EventsFunCore(Xtree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NC)
        flag_contact_calc = DetectContactCore(Xtree, q, qdot, contactpoint, contact_pos_lb, contact_vel_lb, contact_vel_ub,  idcontact, parent, jtype, jaxis, NC)
        qdot_impulse = ImpulsiveDynamicsCore(Xtree, q, qdot, contactpoint, H, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, rankJc)
        H.block_until_ready()
        fqp.block_until_ready()
        xdot.block_until_ready()
        value.block_until_ready()
        flag_contact_calc.block_until_ready()
        

        # flag_contact = DetectContact(model, q, qdot)
        # print(flag_contact)
        qdot_impulse.block_until_ready()
        duarion = time.time() - start_time
        print("Jit compiled time for %s is %s." % (str(flag_contact), duarion))

         


if __name__ == "__main__":
    mdlw = ModelWrapper()
    mdlw.load(os.path.join(CURRENT_PATH, 'whole_max_v1.json'))
    model = mdlw.model
    jit_compiled(model)

    idcontact = model["idcontact"]
    contactpoint = model["contactpoint"]
    q0 = np.array([0, 0, 0.5, 0.0, 0, 0,
        0, 0.5, -0.8,  # fr
        0, 0.5, -0.8,  # fl
        0, 0.5, -0.8,  # br
        0, 0.5, -0.8]) # bl

    q0 = q0.reshape(-1, 1)
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


    

    plt.ion()
    plt.figure()
    fig = plt.gcf()
    ax = Axes3D(fig)  
    ax = plt.gca()
    ax.clear()
    PlotModel(model, q0, ax)
    ax.view_init(elev=0,azim=-90)
    ax.set_xlabel('X')
    ax.set_xlim(-0.3, -0.3+0.6)
    ax.set_ylabel('Y')
    ax.set_ylim(-0.15, -0.15+0.6)
    ax.set_zlabel('Z')
    ax.set_zlim(-0.1, -0.1+0.6)
    plt.pause(0.001)
    plt.show()

    for i in range(1000):
        print(i)
        u = kp * (q0[6:18] - xk[6:18]) + kd * (qd0[6:18] - xk[24:36])
        xk, contact_force = StateFunODE(model, xk.flatten(), u.flatten(), T)
        xk = xk.reshape(-1, 1)
        xksv.append(xk)
        ax.clear()
        PlotModel(model, xk[0:18], ax)
        PlotContactForce(model, xk[0:18], contact_force["fc"], contact_force["fcqp"], contact_force["fcpd"], 'fcqp', ax)
        ax.view_init(elev=0,azim=-90)
        ax.set_xlabel('X')
        ax.set_xlim(-0.3, -0.3+0.6)
        ax.set_ylabel('Y')
        ax.set_ylim(-0.15, -0.15+0.6)
        ax.set_zlabel('Z')
        ax.set_zlim(-0.1, -0.1+0.6)
        ax.set_title('Frame')
        plt.pause(0.001)
        plt.show()


    plt.ioff()