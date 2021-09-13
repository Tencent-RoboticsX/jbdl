from jbdl.experimental.envs.hopper_env import Hopper, DEFAULT_PURE_HOPPER_PARAMS
import jax.numpy as jnp


env = Hopper(render=True)
action = jnp.zeros((3,))
for i in range(1000):
    next_state, reward, done, info = env.step(action)
    env.reset_render_state()
    if done:
        print(i, done)
        env.reset(*DEFAULT_PURE_HOPPER_PARAMS)


