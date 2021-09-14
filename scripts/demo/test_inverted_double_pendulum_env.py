import jax.numpy as jnp
from jbdl.envs.inverted_double_pendulum_env import InvertedDoublePendulum, DEFAULT_PURE_INVERTED_DOUBLE_PENDULUM_PARAMS


env = InvertedDoublePendulum(render=True)
action = jnp.zeros((1, ))
for i in range(1000):
    next_state, reward, done, _ = env.step(action)
    env.reset_render_state()
    if done:
        print(i, "done")
        env.reset(*DEFAULT_PURE_INVERTED_DOUBLE_PENDULUM_PARAMS)


env = InvertedDoublePendulum(batch_size=2, render=False)
action = jnp.zeros((2, 1))
print(env.state)
next_state, reward, done, _ = env.step(action)
print(next_state)