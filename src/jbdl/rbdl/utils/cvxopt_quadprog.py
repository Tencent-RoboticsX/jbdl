import numpy as np
import cvxopt

def cvxopt_quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None, options={'show_progress': False}):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    n_var = H.shape[1]

    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')

    if L is not None or k is not None or lb is not None or ub is not None:
        if L is None:
            L = np.empty([0, n_var])
        if k is None:
            k = np.empty([0, 1])

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
    return np.array(sol['x']), np.array(sol['y']), np.array(sol['z']), str(sol['status'])