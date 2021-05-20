from inspect import stack
import operator
from os import stat
from re import X
from typing import Tuple
from jax.api_util import argnums_partial
import numpy as np
from jbdl.rbdl import contact
from jbdl.rbdl.contact import calc_contact_jacobian
from jbdl.rbdl.contact import calc_contact_jdot_qdot
from jbdl.rbdl.contact import calc_contact_force_pd
from jbdl.rbdl.contact import get_contact_force
from jbdl.rbdl.contact import calc_contact_jacobian_core
from jbdl.rbdl.contact.calc_contact_jacobian import calc_contact_jacobian_extend_core
from jbdl.rbdl.contact.calc_contact_jdot_qdot import calc_contact_jdot_qdot_extend_core
from jbdl.rbdl.contact import calc_contact_jdot_qdot_core
import jax.numpy as jnp
import jax
from jax import jacfwd
from jbdl.rbdl.utils import xyz2int
from jax import lax
from jbdl.rbdl.utils import cvxopt_quadprog
# from jbdl.rbdl.utils.lcp_quadprog import lcp_prim
from jbdl.experimental.custom_ops.lcp import lcp
import cvxopt

def get_A(mu, nf):
    A = jnp.empty([])
    if nf == 2:  
        tx = jnp.array([1, 0])
        tz = jnp.array([0, 1])
        A = jnp.vstack([-mu * tz + tx,
                        -mu * tz - tx,
                        -tz,
                        tz])
                    
    if nf == 3:
        tx = jnp.array([1, 0, 0])
        ty = jnp.array([0, 1, 0])
        tz = jnp.array([0, 0, 1])

        A = jnp.vstack([
            -mu * tz + tx,
            -mu * tz - tx,
            -mu * tz + ty,
            -mu * tz - ty,
            -tz,
            tz
        ])
    return A

def get_b(fzlb, fzub, nf):
    b = jnp.empty([])
    if nf == 2:
        b = jnp.array([0.0, 0.0, -fzlb, fzub])
    if nf == 3:
        b = jnp.array([0.0, 0.0, 0.0, 0.0, -fzlb, fzub])
    return b

def get_zero_A(nf):
    A = jnp.empty([])
    if nf == 2:  
        tx = jnp.array([1.0, 0.0])
        tz = jnp.array([0.0, 1.0])
        A = jnp.vstack([-tx,
                         tx,
                        -tz,
                        tz])
                    
    if nf == 3:
        tx = jnp.array([1.0, 0.0, 0.0])
        ty = jnp.array([0.0, 1.0, 0.0])
        tz = jnp.array([0.0, 0.0, 1.0])

        A = jnp.vstack([
            -tx,
            +tx,
            -ty,
            +ty,
            -tz,
            tz
        ])
    return A

def get_zero_b(nf):
    b = jnp.empty([])
    if nf == 2:
        b = jnp.zeros((4,))
    if nf == 3:
        b = jnp.zeros((6,))
    return b

def get_block_diag_A(flag_contact, mu, nf):
    def f(carry, x):
        new_x = lax.cond(
            x,
            lambda _: get_A(mu, nf),
            lambda _: get_zero_A(nf),
            None
        )
        return carry, new_x

    _, seq_A = lax.scan(f, None, flag_contact)
    block_diag_A = jax.scipy.linalg.block_diag(*seq_A)

    return block_diag_A

def get_stack_b(flag_contact, fzlb, fzub,  nf):
    def f(carry, x):
        new_x = lax.cond(
            x,
            lambda _: get_b(fzlb, fzub, nf),
            lambda _: get_zero_b(nf),
            None
        )
        return carry, new_x

    _, seq_b = lax.scan(f, None, flag_contact)
    stack_b = jnp.reshape(jnp.hstack(seq_b), (-1, 1))
    # stack_b = jnp.reshape(stack_b, (-1, 1))
    # block_diag_A = jax.scipy.linalg.block_diag(*seq_A)

    return stack_b






@jax.custom_jvp
def lcp_quadprog(H, f, L, k, lb, ub):
    cvxopt_qp_H = np.asfarray(H)
    cvxopt_qp_f = np.asfarray(f)
    cvxopt_qp_L = np.asfarray(L)
    cvxopt_qp_k = np.asfarray(k)
    cvxopt_qp_lb = np.asfarray(lb)
    cvxopt_qp_ub = np.asfarray(ub)

    x, _, _, status = cvxopt_quadprog(cvxopt_qp_H, cvxopt_qp_f, L=cvxopt_qp_L, k=cvxopt_qp_k, lb=cvxopt_qp_lb, ub=cvxopt_qp_ub)
    if status != 'optimal':
        print('QP solve failed: status = %', status)
    return x

def lcp_kkt(x, z,  H, f, L, k, lb, ub):
    n_var = H.shape[1]

    L = jnp.vstack([L, -np.eye(n_var)])
    k = jnp.vstack([k, -lb])
    L = jnp.vstack([L, np.eye(n_var)])
    k = jnp.vstack([k, ub])

    lagrange = jnp.matmul(H, x) + f + jnp.matmul(jnp.transpose(L),  z)
    inequality = (jnp.matmul(L, x) - k) * z
    kkt = jnp.vstack([lagrange, inequality])
    return kkt


@lcp_quadprog.defjvp
def lcp_quadprog_jvp(primals, tangents):
    H, f, L, k, lb, ub = primals
    H_dot, f_dot, L_dot, k_dot, lb_dot, ub_dot = tangents

    cvxopt_qp_H = np.asfarray(H)
    cvxopt_qp_f = np.asfarray(f)
    cvxopt_qp_L = np.asfarray(L)
    cvxopt_qp_k = np.asfarray(k)
    cvxopt_qp_lb = np.asfarray(lb)
    cvxopt_qp_ub = np.asfarray(ub)

    n_var = H.shape[1]
    x_star, y_star, z_star, status = cvxopt_quadprog(cvxopt_qp_H, cvxopt_qp_f, L=cvxopt_qp_L, k=cvxopt_qp_k, lb=cvxopt_qp_lb, ub=cvxopt_qp_ub)
    if status != 'optimal':
        print('QP solve failed: status = %', status)
    dkkt2dx = jacfwd(lcp_kkt, argnums=0)(x_star, z_star,  H, f, L, k, lb, ub)
    dkkt2dz = jacfwd(lcp_kkt, argnums=1)(x_star, z_star,  H, f, L, k, lb, ub)
   
    dkkt2dH = jacfwd(lcp_kkt, argnums=2)(x_star, z_star,  H, f, L, k, lb, ub)
    dkkt2df = jacfwd(lcp_kkt, argnums=3)(x_star, z_star,  H, f, L, k, lb, ub)
    dkkt2dL = jacfwd(lcp_kkt, argnums=4)(x_star, z_star,  H, f, L, k, lb, ub)
    dkkt2dk = jacfwd(lcp_kkt, argnums=5)(x_star, z_star,  H, f, L, k, lb, ub)
    dkkt2dlb = jacfwd(lcp_kkt, argnums=6)(x_star, z_star,  H, f, L, k, lb, ub)
    dkkt2dub = jacfwd(lcp_kkt, argnums=7)(x_star, z_star,  H, f, L, k, lb, ub)
   
    dkkt2dxz = jnp.concatenate([dkkt2dx, dkkt2dz], axis=2)
    dkkt2dxz = jnp.transpose(dkkt2dxz, [3, 1, 0, 2])
    dkkt2dH = jnp.transpose(dkkt2dH, [2, 3, 0, 1])
    dkkt2df = jnp.transpose(dkkt2df, [2, 3, 0, 1])
    dkkt2dL = jnp.transpose(dkkt2dL, [2, 3, 0, 1])
    dkkt2dk = jnp.transpose(dkkt2dk, [2, 3, 0, 1])
    dkkt2dlb = jnp.transpose(dkkt2dlb, [2, 3, 0, 1])
    dkkt2dub = jnp.transpose(dkkt2dub, [2, 3, 0, 1])

    dxz2dH = -jnp.linalg.solve(dkkt2dxz, dkkt2dH)
    dxz2dH = jnp.transpose(dxz2dH, [2, 3, 0, 1])
    dxz2dH = dxz2dH[0:n_var, ...]
    dxz2df =  -jnp.linalg.solve(dkkt2dxz, dkkt2df)
    dxz2df = jnp.transpose(dxz2df, [2, 3, 0, 1])
    dxz2df = dxz2df[0:n_var, ...]
    dxz2dL = -jnp.linalg.solve(dkkt2dxz, dkkt2dL)
    dxz2dL = jnp.transpose(dxz2dL, [2, 3, 0, 1])
    dxz2dL = dxz2dL[0:n_var, ...]
    dxz2dk = -jnp.linalg.solve(dkkt2dxz, dkkt2dk)
    dxz2dk = jnp.transpose(dxz2dk, [2, 3, 0, 1])
    dxz2dk = dxz2dk[0:n_var, ...]
    dxz2dlb = -jnp.linalg.solve(dkkt2dxz, dkkt2dlb)
    dxz2dlb = jnp.transpose(dxz2dlb, [2, 3, 0, 1])
    dxz2dlb = dxz2dlb[0:n_var, ...]
    dxz2dub = -jnp.linalg.solve(dkkt2dxz, dkkt2dub)
    dxz2dub = jnp.transpose(dxz2dub, [2, 3, 0, 1])
    dxz2dub = dxz2dub[0:n_var, ...]

    diff_H = jnp.sum(dxz2dH * H_dot, axis=(-2, -1))
    diff_f = jnp.sum(dxz2df * f_dot, axis=(-2, -1))
    diff_L = jnp.sum(dxz2dL * L_dot, axis=(-2, -1))
    diff_k = jnp.sum(dxz2dk * k_dot, axis=(-2, -1))
    diff_lb = jnp.sum(dxz2dlb * lb_dot, axis=(-2, -1))
    diff_ub = jnp.sum(dxz2dub * ub_dot, axis=(-2, -1))


    return x_star, diff_H + diff_f + diff_L + diff_k + diff_lb + diff_ub







def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None, options={'show_progress': False}):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    n_var = H.shape[1]

    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')

    if L is not None or k is not None:
        assert(k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])

        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])

        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if Aeq is not None or beq is not None:
        assert(Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')

    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq, options=options)
    return np.array(sol['x']), str(sol['status'])


def solve_contact_lcp_extend_core(Xtree, q, qdot, contactpoint, H, tau, C, contact_force_lb, contact_force_ub,\
    idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, ncp, mu):

    Jc = calc_contact_jacobian_extend_core(Xtree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
    JcdotQdot = calc_contact_jdot_qdot_extend_core(Xtree, q, qdot, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
    contact_force_lb = jnp.reshape(contact_force_lb, (-1,))
    contact_force_ub = jnp.reshape(contact_force_ub, (-1,))
    if nf == 2:
        contact_force_lb = jnp.array([contact_force_lb[0], contact_force_lb[2]])
        contact_force_ub = jnp.array([contact_force_ub[0], contact_force_ub[2]])
    tau = jnp.reshape(tau, (-1, 1))
    C = jnp.reshape(C, (-1, 1))
    M = jnp.matmul(Jc, jnp.linalg.solve(H, jnp.transpose(Jc)))
    d0 = jnp.matmul(Jc, jnp.linalg.solve(H, tau - C))
    d = jnp.add(d0, JcdotQdot)



    ncd = 2 + 2 * (nf -1)

    A = jnp.zeros((ncd * NC, nf * NC))
    b = jnp.zeros((ncd * NC, 1))
    lb = jnp.zeros((nf * NC, 1))
    ub = jnp.zeros((nf * NC, 1))

    A = get_block_diag_A(flag_contact, mu, nf)
    b = get_stack_b(flag_contact, contact_force_lb[-1], contact_force_ub[-1], nf)
    lb = jnp.reshape(jnp.tile(contact_force_lb, NC), (-1, 1))
    ub = jnp.reshape(jnp.tile(contact_force_ub, NC), (-1, 1))

    # for i in range(NC):
    #     A = A.at[i*ncd:(i+1)*ncd, i*nf:(i+1)*nf].set(get_A(mu, nf))
    #     b = b.at[i*ncd:(i+1)*ncd, 0].set(get_b(contact_force_lb[-1], contact_force_ub[-1], nf))
    #     lb = lb.at[i*nf:(i+1)*nf, 0].set(contact_force_lb)
    #     ub = ub.at[i*nf:(i+1)*nf, 0].set(contact_force_ub)

    # QP optimize contact force in world space
    # M = jnp.eye(nf * NC)
    # d = jnp.zeros((nf * NC, 1))
    M = 0.5 * (M+jnp.transpose(M))
    # H = np.asfarray(H)
    # d = np.asfarray(d)
    # A = np.asfarray(A)
    # b = np.asfarray(b)
    # lb = np.asfarray(lb)
    # ub = np.asfarray(ub)
    # fqp, _, _, _ = cvxopt_quadprog(H, d, A, b, None, None, lb, ub)
    # print(A.shape)
    # print(b.shape)
    fqp = lcp(M, d, A, b, lb, ub)
    # fqp = jnp.zeros(())

    # fqp, _ = quadprog(H, d, A, b, None, None, lb, ub)
    # print(fqp.shape)

    flcp = jnp.matmul(jnp.transpose(Jc), fqp)
    flcp = jnp.reshape(flcp, (-1,))

    return flcp, fqp

    # flcp = jnp.matmul(jnp.transpose(Jc), fqp)

    # flcp = jnp.reshape(flcp, (-1,))

    # return flcp, fqp




def solve_contact_lcp_core(Xtree, q, qdot, contactpoint, H, tau, C, contact_force_lb, contact_force_ub,\
    idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, ncp, mu):

    Jc = calc_contact_jacobian_core(Xtree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
    JcdotQdot = calc_contact_jdot_qdot_core(Xtree, q, qdot, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
    contact_force_lb = jnp.reshape(contact_force_lb, (-1,))
    contact_force_ub = jnp.reshape(contact_force_ub, (-1,))
    if nf == 2:
        contact_force_lb = contact_force_lb[[0, 2]]
        contact_force_ub = contact_force_ub[[0, 2]]
    tau = jnp.reshape(tau, (-1, 1))
    C = jnp.reshape(C, (-1, 1))
    M = jnp.matmul(Jc, jnp.linalg.solve(H, jnp.transpose(Jc)))
    d0 = jnp.matmul(Jc, jnp.linalg.solve(H, tau - C))
    d = jnp.add(d0, JcdotQdot)

    ncd = 2 + 2 * (nf -1)

    A = jnp.zeros((ncd * ncp, nf * ncp))
    b = jnp.zeros((ncd * ncp, 1))
    lb = jnp.zeros((nf * ncp, 1))
    ub = jnp.zeros((nf * ncp, 1))

    for i in range(ncp):
        A = A.at[i*ncd:(i+1)*ncd, i*nf:(i+1)*nf].set(get_A(mu, nf))
        b = b.at[i*ncd:(i+1)*ncd, 0].set(get_b(contact_force_lb[-1], contact_force_ub[-1], nf))
        lb = lb.at[i*nf:(i+1)*nf, 0].set(contact_force_lb)
        ub = ub.at[i*nf:(i+1)*nf, 0].set(contact_force_ub)

    # QP optimize contact force in world space
    M = 0.5 * (M+jnp.transpose(M))
    # H = np.asfarray(H)
    # d = np.asfarray(d)
    # A = np.asfarray(A)
    # b = np.asfarray(b)
    # lb = np.asfarray(lb)
    # ub = np.asfarray(ub)
    # fqp, _, _, _ = cvxopt_quadprog(H, d, A, b, None, None, lb, ub)
    # fqp = lcp_quadprog(H, d, A, b, lb, ub)
    fqp = lcp(M, d, A, b, lb, ub)

    # fqp, _ = quadprog(H, d, A, b, None, None, lb, ub)
    # print(fqp.shape)



    flcp = jnp.matmul(jnp.transpose(Jc), fqp)

    flcp = jnp.reshape(flcp, (-1,))

    return flcp, fqp


        


def solve_contact_lcp(model: dict, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray, \
    flag_contact: np.ndarray, mu: float):
    NC = int(model["NC"])
    NB = int(model["NB"])
    nf = int(model["nf"])
    Xtree = model["Xtree"]
    contactpoint = model["contactpoint"],
    idcontact = tuple(model["idcontact"])
    parent = tuple(model["parent"])
    jtype = tuple(model["jtype"])
    jaxis = xyz2int(model["jaxis"])
    contactpoint = model["contactpoint"]
    flag_contact = flag_contact
    H = model["H"]
    C = model["C"]
    contact_cond = model["contact_cond"]
    contact_force_lb = contact_cond["contact_force_lb"].flatten()
    contact_force_ub = contact_cond["contact_force_ub"].flatten()

    if nf == 2:
        contact_force_lb = contact_force_lb[[0, 2]]
        contact_force_ub = contact_force_ub[[0, 2]]

    ncp = 0
    for i in range(NC):
        if flag_contact[i]!=0:
            ncp = ncp + 1


    flcp, fqp = solve_contact_lcp_core(Xtree, q, qdot, contactpoint, H, tau, C, contact_force_lb, contact_force_ub,\
        idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, ncp, mu)

    fpd = np.zeros((3*NC, 1))
    fc, fcqp, fcpd = get_contact_force(model, fqp, fpd, flag_contact)
    
    return flcp, fqp, fc, fcqp, fcpd

if __name__ == '__main__':
    mu = 0.9
    nf = 3
    fzlb = 0.0
    fzub = 1000
    flag_contact = jnp.array([1, 0, 0, 0])

    from jax import jacfwd, jacrev
    print(get_A(mu, nf))
    print(get_b(fzlb, fzub, nf))
    print(get_zero_A(nf))
    print(get_zero_b(nf))

    print(get_block_diag_A(flag_contact, mu, nf))
    print(get_stack_b(flag_contact, fzlb, fzub, nf))
    # print(jacfwd(get_A, argnums=0)(mu, nf))
    # print(jacrev(get_b, argnums=0)(fzlb, fzub, nf))
    x = jnp.array([0.1, -0.2, 0.3])
    print(jnp.tile(x, 3))





