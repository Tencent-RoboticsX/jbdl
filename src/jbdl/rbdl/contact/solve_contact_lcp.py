import numpy as np
from jbdl.rbdl.contact import get_contact_force
from jbdl.rbdl.contact import calc_contact_jacobian_core
from jbdl.rbdl.contact.calc_contact_jacobian import calc_contact_jacobian_extend_core
from jbdl.rbdl.contact.calc_contact_jdot_qdot import calc_contact_jdot_qdot_extend_core
from jbdl.rbdl.contact import calc_contact_jdot_qdot_core
import jax.numpy as jnp
import jax
from jbdl.rbdl.utils import xyz2int
from jax import lax
from jax.lib import xla_bridge as xb

if xb.get_backend().platform == 'gpu':
    from jbdl.experimental.custom_ops.lcp_gpu import lcp_gpu as lcp
else:
    from jbdl.experimental.custom_ops.lcp import lcp


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


def solve_contact_lcp_extend_core(x_tree, q, qdot, contactpoint, H, tau, C, contact_force_lb, contact_force_ub,\
    idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, ncp, mu):

    Jc = calc_contact_jacobian_extend_core(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf)
    JcdotQdot = calc_contact_jdot_qdot_extend_core(x_tree, q, qdot, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf)
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

    A = jnp.zeros((ncd * nc, nf * nc))
    b = jnp.zeros((ncd * nc, 1))
    lb = jnp.zeros((nf * nc, 1))
    ub = jnp.zeros((nf * nc, 1))

    A = get_block_diag_A(flag_contact, mu, nf)
    b = get_stack_b(flag_contact, contact_force_lb[-1], contact_force_ub[-1], nf)
    lb = jnp.reshape(jnp.tile(contact_force_lb, nc), (-1, 1))
    ub = jnp.reshape(jnp.tile(contact_force_ub, nc), (-1, 1))

    M = 0.5 * (M+jnp.transpose(M))
    fqp, _ = lcp(M, d, A, b, lb, ub)

    flcp = jnp.matmul(jnp.transpose(Jc), fqp)
    flcp = jnp.reshape(flcp, (-1,))

    return flcp, fqp




def solve_contact_lcp_core(x_tree, q, qdot, contactpoint, H, tau, C, contact_force_lb, contact_force_ub,\
    idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, ncp, mu):

    Jc = calc_contact_jacobian_core(x_tree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf)
    JcdotQdot = calc_contact_jdot_qdot_core(x_tree, q, qdot, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf)
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
    fqp = lcp(M, d, A, b, lb, ub)
    flcp = jnp.matmul(jnp.transpose(Jc), fqp)
    flcp = jnp.reshape(flcp, (-1,))

    return flcp, fqp


        


def solve_contact_lcp(model: dict, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray, \
    flag_contact: np.ndarray, mu: float):
    nc = int(model["nc"])
    nb = int(model["nb"])
    nf = int(model["nf"])
    x_tree = model["x_tree"]
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
    for i in range(nc):
        if flag_contact[i]!=0:
            ncp = ncp + 1


    flcp, fqp = solve_contact_lcp_core(x_tree, q, qdot, contactpoint, H, tau, C, contact_force_lb, contact_force_ub,\
        idcontact, flag_contact, parent, jtype, jaxis, nb, nc, nf, ncp, mu)

    fpd = np.zeros((3*nc, 1))
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





