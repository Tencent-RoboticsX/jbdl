from jbdl.envs.cart_pole_env import CartPole, DEFAULT_PURE_CART_POLE_PARAMS
import jax.numpy as jnp
import numpy as np


M_CART = 1.0
M_POLE = 0.1
HALF_POLE_LENGTH = 0.5
POLE_IC_PARAMS = jnp.zeros((6,))
# JOINT_DAMPING_PARAMS = jnp.array([0.0, 0.0])
JOINT_DAMPING_PARAMS = jnp.array([0.7, 0.7])
PURE_CART_POLE_PARAMS = (
    M_CART, M_POLE, HALF_POLE_LENGTH, POLE_IC_PARAMS, JOINT_DAMPING_PARAMS)

env = CartPole(pure_cart_pole_params=PURE_CART_POLE_PARAMS,
               render=True, render_engine_name="xmirror")
action = jnp.zeros((1,))
# action = np.zeros((2,1))
print(env.state)
for i in range(1000):
    next_state, reward, done, _ = env.step(action)
    env.reset_render_state()

    # if done:
    #     print(i, "done")
    #     env.reset(*DEFAULT_PURE_CART_POLE_PARAMS)
