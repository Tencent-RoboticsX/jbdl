from jbdl.envs.cart_pole_env import CartPole, DEFAULT_PURE_CART_POLE_PARAMS
import jax.numpy as jnp
import numpy  as np

env = CartPole(render=True)
action = np.zeros((1,)) 
print(env.state)
for i in range(10000):
    next_state, reward, done, _ = env.step(action)
    env.reset_render_state()

    if done:
        print("done")
        env.reset(*DEFAULT_PURE_CART_POLE_PARAMS)


