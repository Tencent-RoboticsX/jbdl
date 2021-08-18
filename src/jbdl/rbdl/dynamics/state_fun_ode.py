from matplotlib.pyplot import flag
import numpy as np
from jbdl.rbdl.dynamics import composite_rigid_body_algorithm, composite_rigid_body_algorithm_core
from jbdl.rbdl.dynamics import inverse_dynamics, inverse_dynamics_core
from numpy.linalg import inv
from jbdl.rbdl.contact import detect_contact
from jbdl.rbdl.contact import calc_contact_force_direct, calc_contact_force_direct_core
from jbdl.rbdl.dynamics import forward_dynamics, forward_dynamics_core
from jbdl.rbdl.kinematics import calc_body_to_base_coordinates, calc_body_to_base_coordinates_core
from jbdl.rbdl.contact import solve_contact_lcp_core
from jbdl.rbdl.contact import impulsive_dynamics
from jbdl.rbdl.contact import solve_contact_simple_lcp, solve_contact_simple_lcp_core
from jbdl.rbdl.contact.solve_contact_lcp import solve_contact_lcp_extend_core
from scipy.integrate import solve_ivp
import jax.numpy as jnp
from jbdl.rbdl.contact import get_contact_force
from jax.api import jit
from functools import partial
from jbdl.rbdl.utils import xyz2int, calc_rank_jc
from jax import lax

# @partial(jit, static_argnums=(7, 8, 9, 10, 11, 12, 13, 14, 15))
def dynamics_fun_core(x_tree, I, q, qdot, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub,\
    idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, rankJc, ncp, mu):
    H =  composite_rigid_body_algorithm_core(x_tree, I, parent, jtype, jaxis, nb, q)
    C =  inverse_dynamics_core(x_tree, I, parent, jtype, jaxis, nb, q, qdot, jnp.zeros_like(q), a_grav)
    lam = jnp.zeros((nb, ))
    fqp = jnp.zeros((rankJc, 1))




    if np.sum(flag_contact) !=0: 
        lam, fqp = solve_contact_lcp_core(x_tree, q, qdot, contactpoint, H, tau, C, contact_force_lb, contact_force_ub,\
            idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, ncp, mu)

        # lam, fqp = solve_contact_simple_lcp_core(x_tree, q, qdot, contactpoint, H, tau, C, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf)
        # lam, fqp = calc_contact_force_direct_core(x_tree, q, qdot, contactpoint, H, tau, C, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf)

    ttau = tau + lam
    qddot = forward_dynamics_core(x_tree, I, parent, jtype, jaxis, nb, q, qdot, ttau, a_grav)
    # print("========")
    # print(qddot)
    xdot = jnp.hstack([qdot, qddot])
    return xdot, fqp, H

# @partial(jit, static_argnums=(7, 8, 9, 10, 11, 12, 13, 14, 15))
def dynamics_fun_extend_core(x_tree, I, q, qdot, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub,\
    idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, ncp, mu):
    H =  composite_rigid_body_algorithm_core(x_tree, I, parent, jtype, jaxis, nb, q)
    C =  inverse_dynamics_core(x_tree, I, parent, jtype, jaxis, nb, q, qdot, jnp.zeros_like(q), a_grav)

    lam, fqp = lax.cond(
        jnp.sum(flag_contact),
        lambda _: solve_contact_lcp_extend_core(x_tree, q, qdot, contactpoint, H, tau, C, contact_force_lb, contact_force_ub,\
            idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, ncp, mu), 
        lambda _: (jnp.zeros((nb,)), jnp.zeros((nc * nf, 1))),
        operand=None
    )
    # print(lam)

    # lam, fqp = lax.cond(
    #     jnp.sum(flag_contact),
    #     lambda _: (jnp.zeros((nb,)), jnp.zeros((nc * nf, 1))), 
    #     lambda _: (jnp.zeros((nb,)), jnp.zeros((nc * nf, 1))),
    #     operand=None
    # )

    # lam, fqp = (jnp.zeros((nb,)), jnp.zeros((nc * nf, 1)))

    # if jnp.sum(flag_contact):
    #     lam, fqp = solve_contact_lcp_extend_core(x_tree, q, qdot, contactpoint, H, tau, C, contact_force_lb, contact_force_ub,\
    #         idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, ncp, mu)
    # else:
    #     lam = jnp.zeros((nb,))
    #     fqp = jnp.zeros((nc * nf, 1))

    # solve_contact_lcp_extend_core(x_tree, q, qdot, contactpoint, H, tau, C, contact_force_lb, contact_force_ub,\
    #         idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, ncp, mu)


    # if np.sum(flag_contact) !=0: 
    #     lam, fqp = solve_contact_lcp_core(x_tree, q, qdot, contactpoint, H, tau, C, contact_force_lb, contact_force_ub,\
    #         idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, ncp, mu)

        # lam, fqp = solve_contact_simple_lcp_core(x_tree, q, qdot, contactpoint, H, tau, C, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf)
        # lam, fqp = calc_contact_force_direct_core(x_tree, q, qdot, contactpoint, H, tau, C, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf)

    ttau = tau + lam
    qddot = forward_dynamics_core(x_tree, I, parent, jtype, jaxis, nb, q, qdot, ttau, a_grav)
    # print("========")
    # print(qddot)
    xdot = jnp.hstack([qdot, qddot])
    return xdot, fqp, H

def dynamics_fun(t: float, X: np.ndarray, model: dict, contact_force: dict)->np.ndarray:
    # print(X.shape)
    
    nc = int(model["nc"])
    nb = int(model["nb"])
    nf = int(model["nf"])

    q = X[0:nb]
    qdot = X[nb: 2 * nb]
    tau = model["tau"]

    # nc = int(model["nc"])
    # nb = int(model["nb"])
    # nf = int(model["nf"])
    contact_cond = model["contact_cond"]
    x_tree = model["x_tree"]
    contactpoint = model["contactpoint"],
    idcontact = tuple(model["idcontact"])
    parent = tuple(model["parent"])
    jtype = tuple(model["jtype"])
    jaxis = xyz2int(model["jaxis"])
    contactpoint = model["contactpoint"]
    I = model["I"]
    a_grav = model["a_grav"]
    mu = 0.9
    contact_force_lb = contact_cond["contact_force_lb"]
    contact_force_ub = contact_cond["contact_force_ub"]





    # Calculate flag_contact
    flag_contact = detect_contact(model, q, qdot)
    rankJc = int(np.sum( [1 for item in flag_contact if item != 0]) * model["nf"])
    ncp = 0
    for i in range(nc):
        if flag_contact[i]!=0:
            ncp = ncp + 1




    # Dynamics Function Core
    # print("111111111111111111111")
    # print(flag_contact)
    xdot, fqp, H = dynamics_fun_core(x_tree, I, q, qdot, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub, \
        idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, rankJc, ncp, mu)
    model["H"] = H
    # Calculate contact force fot plotting.
    fc = np.zeros((3*nc, 1))
    fcqp = np.zeros((3*nc, 1))  
    fcpd = np.zeros((3*nc, 1))

    if np.sum(flag_contact) !=0: 
        fpd = np.zeros((3*nc, 1))
        fc, fcqp, fcpd = get_contact_force(model, fqp, fpd, flag_contact)

    contact_force["fc"] = fc
    contact_force["fcqp"] = fcqp
    contact_force["fcpd"] = fcpd

    return xdot

@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8))
def events_fun_core(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nc):
    value = jnp.ones((nc,))
    for i in range(nc):
        if flag_contact[i]==2: # Impact
            # Calculate foot height 
            endpos = calc_body_to_base_coordinates_core(x_tree, parent, jtype, jaxis, idcontact[i], q, contactpoint[i])
            value = value.at[i].set(endpos[2, 0])
    return value

@partial(jit, static_argnums=(3, 4, 5, 6, 7))
def calc_contact_point_to_base_cooridnates_core(x_tree, q, contactpoint, idcontact, parent, jtype, jaxis, nc):
    value = jnp.zeros((nc,))
    for i in range(nc):
        endpos = calc_body_to_base_coordinates_core(x_tree, parent, jtype, jaxis, idcontact[i], q, contactpoint[i])
        value = value.at[i].set(endpos[2, 0])

    return value

@partial(jit, static_argnums=(3, 5, 6, 7, 8))
def events_fun_extend_core(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nc):
    event_value = lax.cond(
        jnp.any(jnp.logical_not(flag_contact-2.0)),
        lambda _: jnp.min(calc_contact_point_to_base_cooridnates_core(x_tree, q, contactpoint, idcontact, parent, jtype, jaxis, nc)),
        lambda _: 1.0,
        operand = None,
    )
    return event_value


        




def events_fun(t: float, x: np.ndarray, model: dict, contact_force: dict=dict()):
    # print("6666666666666666666666")
    nb = int(model["nb"])
    nc = int(model["nc"])
   
    # Get q qdot tau
    q = x[0: nb]
    qdot = x[nb: 2*nb]

    x_tree = model["x_tree"]
    idcontact = tuple(model["idcontact"])
    contactpoint = model["contactpoint"]
    # print(contactpoint)
    parent = tuple(model["parent"])
    jtype = tuple(model["jtype"])
    jaxis = xyz2int(model["jaxis"])


    
    # print("77777777777777777777777777")
    # Detect contact
    flag_contact = detect_contact(model, q, qdot)
    # print("In EventsFun!!!")
    # print(flag_contact)
    # print("8888888888888888888")

    value = events_fun_core(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nc)

    # print("9999999999999999999")

    return value


def state_fun_ode(model: dict, xk: np.ndarray, uk: np.ndarray, T: float):

    nb = int(model["nb"])
    nc = int(model["nc"])
    ST = model["ST"]

    # Get q qdot tau
    q = xk[0: nb]
    qdot = xk[nb: 2*nb]
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
    
    # event_list = [lambda t, x, model,  contact_force:  hit_ground(t, x, model, contact_force, idx=i) for i in range(nc)]

    def event(t, x, model, contact_force):
        res = events_fun(t, x, model, contact_force)
        res = np.min(res)
        return res

    event.terminal = True
    event.direction = -1


    # for event in event_list:
    #     event.terminal = True
    #     event.direction = -1

    # t_events = [np.array([], dtype=float) for i in range(nc)]
    
    status = -1

    contact_force = dict()

    while status != 0:
        # print("00000000000000000000000")
        # ODE calculate 
        sol = solve_ivp(dynamics_fun, tspan, x0.flatten(), method='RK45', events=event, \
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
            q = xe[0:nb]
            qdot = xe[nb:2* nb]

            # print("33333333333333333333333")
            # Detect contact
            flag_contact = detect_contact(model, q, qdot)
            # print(flag_contact)

            # Impact dynamics
            # print("4444444444444444444444444")
            # print(flag_contact)
            qdot_impulse = impulsive_dynamics(model, q, qdot, flag_contact);  
            qdot_impulse = qdot_impulse.flatten()

            # Update initial state
            x0 = np.hstack([q, qdot_impulse])
            tspan = (te, tf)
            # print("5555555555555555555555555")

    xk = sol.y[:, -1]
    return xk, contact_force

