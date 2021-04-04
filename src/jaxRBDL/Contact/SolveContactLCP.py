from os import stat
import numpy as np
import cvxopt
from jaxRBDL.Contact import calc_contact_jacobian
from jaxRBDL.Contact import calc_contact_jdot_qdot
from jaxRBDL.Contact.CalcContactForcePD import CalcContactForcePD
from jaxRBDL.Contact.GetContactForce import GetContactForce

def GetA(mu: float, nf: int)->np.ndarray:
    """Set friction constraint.

    Args:
        mu (float): riction coeeficient,should between [0, 1], 0 means no contact.

    Returns:
        np.ndarray: [description]
    """
    A = np.empty([])
    if nf == 2:  
        tx = np.array([1, 0])
        tz = np.array([0, 1])
        A = np.vstack([-mu * tz + tx,
                        -mu * tz - tx,
                        -tz,
                        tz])
        A = np.asfarray(A)
                    
    if nf == 3:
        tx = np.array([1, 0, 0])
        ty = np.array([0, 1, 0])
        tz = np.array([0, 0, 1])

        A = np.vstack([
            -mu * tz + tx,
            -mu * tz - tx,
            -mu * tz + ty,
            -mu * tz - ty,
            -tz,
            tz
        ])
        A = np.asfarray(A)
    return A

def Getb(fzlb: float, fzub: float, nf: int)->np.ndarray:
    """[summary]

    Args:
        fzlb (float): lower bound of fz
        fzub (float): upper bound of fz
        nf (int): number of force freedom

    Returns:
        np.ndarray: [description]
    """
    b = np.empty([])
    if nf == 2:
        b = np.array([0.0, 0.0, -fzlb, fzub])
    if nf == 3:
        b = np.array([0.0, 0.0, 0.0, 0.0, -fzlb, fzub])
    return b




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





        


def SolveContactLCP(model: dict, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray, \
    flag_contact: np.ndarray, mu: float):
    
    NC = int(model["NC"])
    nf = int(model["nf"])
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
    
    # Calculate contact force
    Jc = calc_contact_jacobian(model, q, flag_contact)
    JcdotQdot = calc_contact_jdot_qdot(model, q, qdot, flag_contact)

    M = np.matmul(np.matmul(Jc, model["Hinv"]), np.transpose(Jc))
    tau = tau.reshape(*(-1, 1))
    d0 = np.matmul(np.matmul(Jc, model["Hinv"]), tau - model["C"])
    d = np.add(d0, JcdotQdot)
    # R = np.eye(M.shape[0]) 

    # Set contact force inequality constraint
    ncd = 2 + 2* (nf -1)
    A = np.zeros((ncd * ncp, nf * ncp))
    b = np.zeros((ncd * ncp, 1))
    lb = np.zeros((nf * ncp, 1))
    ub = np.zeros((nf * ncp, 1))

    for i in range(ncp):
        A[i*ncd:(i+1)*ncd, i*nf:(i+1)*nf] = GetA(mu, nf)
        b[i*ncd:(i+1)*ncd, 0] = Getb(contact_force_lb[-1], contact_force_ub[-1], nf)
        lb[i*nf:(i+1)*nf, 0] = contact_force_lb
        ub[i*nf:(i+1)*nf, 0] = contact_force_ub

    # QP optimize contact force in world space
    H = np.asfarray(0.5 * (M+M.transpose()))
    d = np.asfarray(d)
    fqp, status = quadprog(H, d, A, b, None, None, lb, ub)
    
    if status != 'optimal':
        print('QP solve failed: status = %', status)

    # Calculate contact force from PD controller
    # contact_force_kp = np.array([10000.0, 10000.0, 10000.0])
    # contact_force_kd = np.array([1000.0, 1000.0, 1000.0])

    # Calculate contact force from PD controller
    fpd = CalcContactForcePD(model, q, qdot, flag_contact)

    # Translate contact force in joint space
    flcp = np.matmul(np.transpose(Jc), fqp)
    # flcp = Jc' * (fqp + fpd);

    # Get contact force for plot
    fc, fcqp, fcpd = GetContactForce(model, fqp, fpd, flag_contact)  

    return flcp, fqp, fc, fcqp, fcpd





