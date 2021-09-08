from jbdl.envs.hopper_env import Hopper, DEFAULT_PURE_HOPPER_PARAMS
import jax.numpy as jnp
import numpy as np

env = Hopper(render=True)
# action = jnp.zeros((6,))
action = jnp.zeros((3,))
print(env.state)
for i in range(1000):
    print(i)
    next_state, reward, done, _ = env.step(action)
    print(env.state)
    env.reset_render_state()