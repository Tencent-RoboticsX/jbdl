import jax.numpy as jnp
from jbdl.envs.inverted_double_pendulum_env import InvertedDoublePendulum, DEFAULT_PURE_INVERTED_DOUBLE_PENDULUM_PARAMS


env = InvertedDoublePendulum(render=True)
print(env.state)
# action = jnp.zeros((2, 1))
action = jnp.zeros((1, ))
# action = jnp.ones((1,))
for i in range(10000):
    next_state, reward, done, _ = env.step(action)
    # print(i, next_state)
    env.reset_render_state()
    # print(done)
    if done:
        print(i, "done")
        env.reset(*DEFAULT_PURE_INVERTED_DOUBLE_PENDULUM_PARAMS)
