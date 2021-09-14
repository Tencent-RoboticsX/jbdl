from jax._src.api import F
from jbdl.envs.cart_pole_env import CartPole
import jax.numpy as jnp
import numpy as np


M_CART = 1.0
M_POLE = 0.1
HALF_POLE_LENGTH = 0.5
POLE_IC_PARAMS = jnp.zeros((6,))
JOINT_DAMPING_PARAMS = jnp.array([0.7, 0.7])
PURE_CART_POLE_PARAMS = (
    M_CART, M_POLE, HALF_POLE_LENGTH, POLE_IC_PARAMS, JOINT_DAMPING_PARAMS)

env = CartPole(pure_cart_pole_params=PURE_CART_POLE_PARAMS,
               render=True, render_engine_name="xmirror")

action = jnp.zeros((1,))
for i in range(1000):
    next_state, reward, done, _ = env.step(action)
    env.reset_render_state()

    if done:
        print(i, "done")
        env.reset(*PURE_CART_POLE_PARAMS)


env = CartPole(batch_size=2, pure_cart_pole_params=PURE_CART_POLE_PARAMS,
               render=False, render_engine_name="xmirror")
print(env.state)
action = jnp.zeros((2, 1))
next_state, reward, done, _ = env.step(action)
print(next_state)