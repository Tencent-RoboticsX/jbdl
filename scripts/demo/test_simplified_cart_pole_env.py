from jbdl.envs.simplified_cart_pole_env import SimplifiedCartPole, DEFAULT_PURE_SIMPLIFIED_CART_POLE_PARAMS
import jax.numpy as jnp
import numpy  as np

env = SimplifiedCartPole(render=True)
action = np.zeros((1,)) 
print(env.state)
for i in range(100):
    next_state, reward, done, _ = env.step(action)
    env.reset_render_state()
    print(next_state, reward, done)
    print(env.state)

    if done:
        env.reset(DEFAULT_PURE_SIMPLIFIED_CART_POLE_PARAMS)


