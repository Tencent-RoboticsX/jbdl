import numpy as np
from jaxRBDL.Dynamics.CompositeRigidBodyAlgorithm import CompositeRigidBodyAlgorithm
from jaxRBDL.Dynamics.InverseDynamics import InverseDynamics
from numpy.linalg import inv
from jaxRBDL.Contact.DetectContact import DetectContact
from jaxRBDL.Contact.CalcContactForceDirect import CalcContactForceDirect
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics
from jaxRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates
from jaxRBDL.Contact.ImpulsiveDynamics import ImpulsiveDynamics
from jaxRBDL.Contact.SolveContactLCP import SolveContactLCP
from scipy.integrate import solve_ivp

def DynamicsFun(t: float, X: np.ndarray, model: dict, contact_cond: dict, contact_force: dict)->np.ndarray:
    # print(X.shape)
    
    NB = int(model["NB"])
    NC = int(model["NC"])
    ST = model["ST"]



    # Get q qdot tau
    q = X[0:NB]
    qdot = X[NB: 2 * NB]
    tau = model["tau"]


    # Calcualte H C 
    model["H"] = CompositeRigidBodyAlgorithm(model, q)
    model["C"] = InverseDynamics(model, q, qdot, np.zeros((NB, 1)))
    model["Hinv"] = inv(model["H"])

    # # Set Contact Conditions.
    # contact_cond = dict()
    # contact_cond["contact_pos_lb"] = np.array([0.0001, 0.0001, 0.0001])
    # contact_cond["contact_pos_ub"] = np.array([0.0001, 0.0001, 0.0001])
    # contact_cond["contact_vel_lb"] = np.array([-0.05, -0.05, -0.05])
    # contact_cond["contact_vel_ub"] = np.array([0.01, 0.01, 0.01])

    # Calculate contact force in joint space
    flag_contact = DetectContact(model, q, qdot, contact_cond)
    # print("In Dynamics!!!")
    # print(flag_contact)
    if np.sum(flag_contact) !=0: 
        # lambda, fqp, fpd] = SolveContactLCP(q, qdot, tau, flag_contact);
        contact_cond = dict()
        contact_cond["contact_force_lb"] = np.array([-1000.0, -1000.0, 0.0])
        contact_cond["contact_force_ub"] = np.array([1000.0, 1000.0, 3000.0])
        lam, fqp, fc, fcqp, fcpd = SolveContactLCP(model, q, qdot, tau, flag_contact, 2, contact_cond, 0.9)
        # lam, fqp, fc, fcqp, fcpd = CalcContactForceDirect(model, q, qdot, tau, flag_contact, 3)
        contact_force["fc"] = fc
        contact_force["fcqp"] = fcqp
        contact_force["fcpd"] = fcpd
    else:
        # print("No Conatact")
        lam = np.zeros((NB, 1))
        contact_force["fc"] = np.zeros((3*NC, 1))
        contact_force["fcqp"] = np.zeros((3*NC, 1))
        contact_force["fcpd"] = np.zeros((3*NC, 1))


    # Forward dynamics
    Tau = tau + lam
    qddot = ForwardDynamics(model, q, qdot, Tau).flatten()

    # Return Xdot
    Xdot = np.asfarray(np.hstack([qdot, qddot]))

    return Xdot

def EventsFun(t: float, X: np.ndarray, model: dict, contact_cond: dict, contact_force: dict=dict()):
    NB = int(model["NB"])
    NC = int(model["NC"])
   
    # Get q qdot tau
    q = X[0: NB]
    qdot = X[NB: 2*NB]
    tau = model["tau"]

    value = np.ones((NC, 1))
    isterminal = np.ones((NC, 1))
    direction = -np.ones((NC, 1))

    try: 
        idcontact = np.squeeze(model["idcontact"], axis=0).astype(int)
        contactpoint = np.squeeze(model["contactpoint"], axis=0)
    except:
        idcontact = model["idcontact"]
        contactpoint = model["contactpoint"]

    # Detect contact
    flag_contact = DetectContact(model, q, qdot, contact_cond)
    # print("In EventsFun!!!")
    # print(flag_contact)
    value_list = []
    for i in range(NC):
        if flag_contact[i]==2: # Impact
            # Calculate foot height 
            endpos = CalcBodyToBaseCoordinates(model, q, idcontact[i], contactpoint[i])
            value[i,0] = endpos[2, 0]

    return value




def StateFunODE(model: dict, xk: np.ndarray, uk: np.ndarray, T: float, contact_cond: dict):

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

    def hit_ground(t, x, model, contact_cond, contact_force, idx=0):
        res = EventsFun(t, x, model, contact_cond, contact_force)
        res = res.flatten()
        return res[idx]
    
    event_list = [lambda t, x, model, contact_cond, contact_force:  hit_ground(t, x, model, contact_cond, contact_force, idx=i) for i in range(NC)]

    for event in event_list:
        event.terminal = True
        event.direction = -1

    # t_events = [np.array([], dtype=float) for i in range(NC)]
    
    status = -1

    contact_force = dict()

    while status != 0:
        # ODE calculate 
        sol = solve_ivp(DynamicsFun, tspan, x0.flatten(), method='RK45', events=event_list, \
            args=(model, contact_cond, contact_force), rtol=1e-3, atol=1e-4)
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

            # Detect contact
            flag_contact = DetectContact(model, q, qdot, contact_cond)

            # Impact dynamics
            qdot_impulse = ImpulsiveDynamics(model, q, qdot, flag_contact, nf=2);  
            qdot_impulse = qdot_impulse.flatten()

            # Update initial state
            x0 = np.hstack([q, qdot_impulse])
            tspan = (te, tf)

    xk = sol.y[:, -1]
    return xk, contact_force

