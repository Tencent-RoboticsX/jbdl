from jbdl.envs.reacher_env import Reacher, DEFAULT_PURE_REACHER_PARAMS
import jax.numpy as jnp
import numpy as np

env = Reacher(render=True)
action = jnp.array([1.0, -1.0])
print(env.state)
for i in range(100):
    print(i)
    next_state, reward, done, _ = env.step(action)
    print(env.state)
    # env.state = jnp.array([jnp.sin(i/100.0), jnp.cos(i/100.0), 0., 0.])
    env.reset_render_state()

    # if done:
    #     print("done")
    #     env.reset(*DEFAULT_PURE_REACHER_PARAMS)
