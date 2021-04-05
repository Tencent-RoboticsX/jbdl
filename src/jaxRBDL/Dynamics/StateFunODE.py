import numpy as np
from jaxRBDL.Dynamics import composite_rigid_body_algorithm, composite_rigid_body_algorithm_core
from jaxRBDL.Dynamics import inverse_dynamics, inverse_dynamics_core
from numpy.linalg import inv
from jaxRBDL.Contact.DetectContact import DetectContact
from jaxRBDL.Contact.CalcContactForceDirect import CalcContactForceDirect
from jaxRBDL.Dynamics import forward_dynamics, forward_dynamics_core
from jaxRBDL.Kinematics import calc_body_to_base_coordinates, calc_body_to_base_coordinates_core
from jaxRBDL.Contact.ImpulsiveDynamics import ImpulsiveDynamics
from jaxRBDL.Contact.SolveContactLCP import SolveContactLCP
from jaxRBDL.Contact.SolveContactSimpleLCP import SolveContactSimpleLCP, SolveContactSimpleLCPCore
from jaxRBDL.Contact.CalcContactForceDirect import CalcContactForceDirectCore
from scipy.integrate import solve_ivp
import jax.numpy as jnp
from jaxRBDL.Contact.GetContactForce import GetContactForce
from jax.api import jit
from functools import partial

# @partial(jit, static_argnums=(7, 8, 9, 10, 11, 12, 13, 14, 15))
def DynamicsFunCore(Xtree, I, q, qdot, contactpoint, tau, a_grav, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, rankJc):
    H =  composite_rigid_body_algorithm_core(Xtree, I, parent, jtype, jaxis, NB, q)
    C =  inverse_dynamics_core(Xtree, I, parent, jtype, jaxis, NB, q, qdot, jnp.zeros_like(q), a_grav)
    lam = jnp.zeros((NB, ))
    fqp = jnp.zeros((rankJc, 1))

    if np.sum(flag_contact) !=0: 
        lam, fqp = SolveContactSimpleLCPCore(Xtree, q, qdot, contactpoint, H, tau, C, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
        # lam, fqp = CalcContactForceDirectCore(Xtree, q, qdot, contactpoint, H, tau, C, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)


    ttau = tau + lam
    qddot = forward_dynamics_core(Xtree, I, parent, jtype, jaxis, NB, q, qdot, ttau, a_grav)
    # print("========")
    # print(qddot)
    xdot = jnp.hstack([qdot, qddot])
    return xdot, fqp, H

def DynamicsFun(t: float, X: np.ndarray, model: dict, contact_force: dict)->np.ndarray:
    # print(X.shape)
    
    NC = int(model["NC"])
    NB = int(model["NB"])
    nf = int(model["nf"])

    q = X[0:NB]
    qdot = X[NB: 2 * NB]
    tau = model["tau"]

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
    I = model["I"]
    a_grav = model["a_grav"]


    # Calculate flag_contact
    flag_contact = DetectContact(model, q, qdot)
    rankJc = int(np.sum( [1 for item in flag_contact if item != 0]) * model["nf"])



    # Dynamics Function Core
    # print("111111111111111111111")
    # print(flag_contact)
    xdot, fqp, H = DynamicsFunCore(Xtree, I, q, qdot, contactpoint, tau, a_grav, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, rankJc)
    model["H"] = H
    # Calculate contact force fot plotting.
    fc = np.zeros((3*NC, 1))
    fcqp = np.zeros((3*NC, 1))  
    fcpd = np.zeros((3*NC, 1))

    if np.sum(flag_contact) !=0: 
        fpd = np.zeros((3*NC, 1))
        fc, fcqp, fcpd = GetContactForce(model, fqp, fpd, flag_contact)

    contact_force["fc"] = fc
    contact_force["fcqp"] = fcqp
    contact_force["fcpd"] = fcpd

    return xdot

@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8))
def EventsFunCore(Xtree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NC):
    value = jnp.ones((NC,))
    for i in range(NC):
        if flag_contact[i]==2: # Impact
            # Calculate foot height 
            endpos = calc_body_to_base_coordinates_core(Xtree, parent, jtype, jaxis, idcontact[i], q, contactpoint[i])
            value = value.at[i].set(endpos[2, 0])
    return value

def EventsFun(t: float, x: np.ndarray, model: dict, contact_force: dict=dict()):
    # print("6666666666666666666666")
    NB = int(model["NB"])
    NC = int(model["NC"])
   
    # Get q qdot tau
    q = x[0: NB]
    qdot = x[NB: 2*NB]

    Xtree = model["Xtree"]
    idcontact = tuple(model["idcontact"])
    contactpoint = model["contactpoint"]
    # print(contactpoint)
    parent = tuple(model["parent"])
    jtype = tuple(model["jtype"])
    jaxis = model["jaxis"]


    
    # print("77777777777777777777777777")
    # Detect contact
    flag_contact = DetectContact(model, q, qdot)
    # print("In EventsFun!!!")
    # print(flag_contact)
    # print("8888888888888888888")

    value = EventsFunCore(Xtree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NC)

    # print("9999999999999999999")

    return value


def StateFunODE(model: dict, xk: np.ndarray, uk: np.ndarray, T: float):

    NB = int(model["NB"])
    NC = int(model["NC"])
    ST = model["ST"]

    # Get q qdot tau
    q = xk[0: NB]
    qdot = xk[NB: 2*NB]
    tau = np.matmul(ST, uk)
    model["tau"] = tau

    # Calculate state vector by ODE
    t0 = 0
    tf = T
    tspan = (t0, tf)

    x0 = np.asfarray(np.hstack([q, qdot]))
    # print(x0.shape)


    te = t0

    # def hit_ground(t, x, model,  contact_force, idx=0):
    #     res = EventsFun(t, x, model, contact_force)
    #     res = res.flatten()
    #     return res[idx]
    
    # event_list = [lambda t, x, model,  contact_force:  hit_ground(t, x, model, contact_force, idx=i) for i in range(NC)]

    def event(t, x, model, contact_force):
        res = EventsFun(t, x, model, contact_force)
        res = np.min(res)
        return res

    event.terminal = True
    event.direction = -1


    # for event in event_list:
    #     event.terminal = True
    #     event.direction = -1

    # t_events = [np.array([], dtype=float) for i in range(NC)]
    
    status = -1

    contact_force = dict()

    while status != 0:
        # print("00000000000000000000000")
        # ODE calculate 
        sol = solve_ivp(DynamicsFun, tspan, x0.flatten(), method='RK45', events=event, \
            args=(model, contact_force), rtol=1e-3, atol=1e-4)
        status = sol.status
        assert status != -1, "Integration Failed!!!"

        if status == 0:
            pass
            # print("The solver successfully reached the end of tspan.")

        if status == 1:
            # print("A termination event occurred")
            t_events = sol.t_events
            te_idx = t_events.index(min(t_events))
            te = float(t_events[te_idx])
            xe = sol.y_events[te_idx].flatten()

            # Get q qdot
            q = xe[0:NB]
            qdot = xe[NB:2* NB]

            # print("33333333333333333333333")
            # Detect contact
            flag_contact = DetectContact(model, q, qdot)
            # print(flag_contact)

            # Impact dynamics
            # print("4444444444444444444444444")
            # print(flag_contact)
            qdot_impulse = ImpulsiveDynamics(model, q, qdot, flag_contact);  
            qdot_impulse = qdot_impulse.flatten()

            # Update initial state
            x0 = np.hstack([q, qdot_impulse])
            tspan = (te, tf)
            # print("5555555555555555555555555")

    xk = sol.y[:, -1]
    return xk, contact_force

