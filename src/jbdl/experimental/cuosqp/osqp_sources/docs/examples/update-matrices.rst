Update matrices
===============


Consider the following QP


.. math::
  \begin{array}{ll}
    \mbox{minimize} & \frac{1}{2} x^T \begin{bmatrix}4 & 1\\ 1 & 2 \end{bmatrix} x + \begin{bmatrix}1 \\ 1\end{bmatrix}^T x \\
    \mbox{subject to} & \begin{bmatrix}1 \\ 0 \\ 0\end{bmatrix} \leq \begin{bmatrix} 1 & 1\\ 1 & 0\\ 0 & 1\end{bmatrix} x \leq \begin{bmatrix}1 \\ 0.7 \\ 0.7\end{bmatrix}
  \end{array}



We show below how to setup and solve the problem.
Then we update the matrices :math:`P` and :math:`A` and solve the updated problem


.. math::
  \begin{array}{ll}
    \mbox{minimize} & \frac{1}{2} x^T \begin{bmatrix}5 & 1.5\\ 1.5 & 1 \end{bmatrix} x + \begin{bmatrix}1 \\ 1\end{bmatrix}^T x \\
    \mbox{subject to} & \begin{bmatrix}1 \\ 0 \\ 0\end{bmatrix} \leq \begin{bmatrix} 1.2 & 1.1\\ 1.5 & 0\\ 0 & 0.8\end{bmatrix} x \leq \begin{bmatrix}1 \\ 0.7 \\ 0.7\end{bmatrix}
  \end{array}
  


Python
------

.. code:: python

    import osqp
    import numpy as np
    from scipy import sparse

    # Define problem data
    P = sparse.csc_matrix([[4, 1], [1, 2]])
    q = np.array([1, 1])
    A = sparse.csc_matrix([[1, 1], [1, 0], [0, 1]])
    l = np.array([1, 0, 0])
    u = np.array([1, 0.7, 0.7])

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, l, u)

    # Solve problem
    res = prob.solve()

    # Update problem
    # NB: Update only upper triangular part of P
    P_new = sparse.csc_matrix([[5, 1.5], [1.5, 1]])
    A_new = sparse.csc_matrix([[1.2, 1.1], [1.5, 0], [0, 0.8]])
    prob.update(Px=sparse.triu(P_new).data, Ax=A_new.data)

    # Solve updated problem
    res = prob.solve()



Matlab
------

.. code:: matlab

    % Define problem data
    P = sparse([4, 1; 1, 2]);
    q = [1; 1];
    A = sparse([1, 1; 1, 0; 0, 1]);
    l = [1; 0; 0];
    u = [1; 0.7; 0.7];

    % Create an OSQP object
    prob = osqp;

    % Setup workspace
    prob.setup(P, q, A, l, u);

    % Solve problem
    res = prob.solve();

    % Update problem
    % NB: Update only upper triangular part of P
    P_new = sparse([5, 1.5; 1.5, 1]);
    A_new = sparse([1.2, 1.1; 1.5, 0; 0, 0.8]);
    prob.update('Px', nonzeros(triu(P_new)), 'Ax', nonzeros(A_new));

    % Solve updated problem
    res = prob.solve();



Julia
------

.. code:: julia

    using OSQP
    using Compat.SparseArrays, Compat.LinearAlgebra

    # Define problem data
    P = sparse([4. 1.; 1. 2.])
    q = [1.; 1.]
    A = sparse([1. 1.; 1. 0.; 0. 1.])
    l = [1.; 0.; 0.]
    u = [1.; 0.7; 0.7]

    # Crate OSQP object
    prob = OSQP.Model()

    # Setup workspace
    OSQP.setup!(prob; P=P, q=q, A=A, l=l, u=u)

    # Solve problem
    results = OSQP.solve!(prob)

    # Update problem
    # NB: Update only upper triangular part of P
    P_new = sparse([5. 1.5; 1.5 1.])
    A_new = sparse([1.2 1.1; 1.5 0.; 0. 0.8])
    OSQP.update!(prob, Px=triu(P_new).nzval, Ax=A_new.nzval)

    # Solve updated problem
    results = OSQP.solve!(prob)



C
-

.. code:: c

    #include "osqp.h"

    int main(int argc, char **argv) {
        /* Load problem data */
        c_float P_x[3]     = {4.0, 1.0, 2.0, };
        c_float P_x_new[3] = {5.0, 1.5, 1.0, };
        c_int   P_nnz      = 3;
        c_int   P_i[3]     = {0, 0, 1, };
        c_int   P_p[3]     = {0, 1, 3, };
        c_float q[2]       = {1.0, 1.0, };
        c_float q_new[2]   = {2.0, 3.0, };
        c_float A_x[4]     = {1.0, 1.0, 1.0, 1.0, };
        c_float A_x_new[4] = {1.2, 1.5, 1.1, 0.8, };
        c_int   A_nnz      = 4;
        c_int   A_i[4]     = {0, 1, 0, 2, };
        c_int   A_p[3]     = {0, 2, 4, };
        c_float l[3]       = {1.0, 0.0, 0.0, };
        c_float l_new[3]   = {2.0, -1.0, -1.0, };
        c_float u[3]       = {1.0, 0.7, 0.7, };
        c_float u_new[3]   = {2.0, 2.5, 2.5, };
        c_int n = 2;
        c_int m = 3;

        /* Exitflag */
        c_int exitflag;

        /* Workspace, settings, matrices */
        OSQPWorkspace *work;
        OSQPSettings *settings;
        csc *P, *A;

        /* Populate matrices */
        P = csc_matrix(n, n, P_nnz, P_x, P_i, P_p);
        A = csc_matrix(m, n, A_nnz, A_x, A_i, A_p);

        /* Set default settings */
        settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
        if (settings) osqp_set_default_settings(settings);

        /* Setup workspace */
        exitflag = osqp_setup(&work, P, q, A, l, u, m, n, settings);

        /* Solve problem */
        osqp_solve(work);

        /* Update problem
           NB: Update only upper triangular part of P
         */
        osqp_update_P(work, P_x_new, OSQP_NULL, 3);
        osqp_update_A(work, A_x_new, OSQP_NULL, 4);

        /* Solve updated problem */
        osqp_solve(work);

        /* Clean workspace */
        osqp_cleanup(work);
        c_free(A);
        c_free(P);
        c_free(settings);

        return exitflag;
    }
