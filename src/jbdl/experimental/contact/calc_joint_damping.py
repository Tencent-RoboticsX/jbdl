import jax
import jax.numpy as jnp


def calc_joint_damping_core(qdot, damping_coef):
    qdot = jnp.reshape(qdot, (-1,))
    damping_coef = jnp.reshape(damping_coef, (-1,))
    tau =  -qdot * damping_coef
    return tau

calc_joint_damping = jax.jit(calc_joint_damping_core)

if __name__ == "__main__":
    qdot = jnp.ones((7,))
    tau = calc_joint_damping_core(qdot, 0.7)
    print(tau)
    print(tau.shape)
    tau = calc_joint_damping(qdot, 0.7)
    print(tau)
    print(tau.shape)
