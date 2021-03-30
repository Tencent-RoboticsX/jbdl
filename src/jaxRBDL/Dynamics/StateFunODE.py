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

def DynamicsFun(t: float, X: np.ndarray, model: dict, contact_force: dict)->np.ndarray:
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



    # Calculate contact force in joint space
    flag_contact = DetectContact(model, q, qdot)
    # print("In Dynamics!!!")
    # print(flag_contact)
    if np.sum(flag_contact) !=0: 
        # lam, fqp, fc, fcqp, fcpd = SolveContactLCP(model, q, qdot, tau, flag_contact, 0.9)
        lam, fqp, fc, fcqp, fcpd = CalcContactForceDirect(model, q, qdot, tau, flag_contact)
        contact_force["fc"] = fc
        contact_force["fcqp"] = fcqp
        contact_force["fcpd"] = fcpd
    else:
        # print("No Conatact")
        lam = np.zeros((NB, 1))
        contact_force["fc"] = np.zeros((3*NC, 1))
        contact_force["fcqp"] = np.zeros((3*NC, 1))
        contact_force["fcpd"] = np.zeros((3*NC, 1))

    print("11111111111111111111")
    # Forward dynamics
    Tau = tau + lam
    qddot = ForwardDynamics(model, q, qdot, Tau).flatten()

    # Return Xdot
    Xdot = np.asfarray(np.hstack([qdot, qddot]))
    print("2222222222222222222222")
    return Xdot

def EventsFun(t: float, X: np.ndarray, model: dict, contact_force: dict=dict()):
    print("6666666666666666666666")
    NB = int(model["NB"])
    NC = int(model["NC"])
   
    # Get q qdot tau
    q = X[0: NB]
    qdot = X[NB: 2*NB]
    tau = model["tau"]

    value = np.ones((NC, 1))
    isterminal = np.ones((NC, 1))
    direction = -np.ones((NC, 1))
    idcontact = model["idcontact"]
    contactpoint = model["contactpoint"]
    
    print("77777777777777777777777777")
    # Detect contact
    flag_contact = DetectContact(model, q, qdot)
    # print("In EventsFun!!!")
    # print(flag_contact)
    print("8888888888888888888")
    value_list = []
    for i in range(NC):
        if flag_contact[i]==2: # Impact
            # Calculate foot height 
            endpos = CalcBodyToBaseCoordinates(model, q, idcontact[i], contactpoint[i])
            value[i,0] = endpos[2, 0]
    print("9999999999999999999")

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
        res = np.min(res.flatten())
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
        print("00000000000000000000000")
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

            print("33333333333333333333333")
            # Detect contact
            flag_contact = DetectContact(model, q, qdot)

            # Impact dynamics
            print("4444444444444444444444444")
            print(flag_contact)
            qdot_impulse = ImpulsiveDynamics(model, q, qdot, flag_contact);  
            qdot_impulse = qdot_impulse.flatten()

            # Update initial state
            x0 = np.hstack([q, qdot_impulse])
            tspan = (te, tf)
            print("5555555555555555555555555")

    xk = sol.y[:, -1]
    return xk, contact_force

