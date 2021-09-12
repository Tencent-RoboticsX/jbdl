from jbdl.experimental.envs.hopper_env import Hopper, DEFAULT_PURE_HOPPER_PARAMS
import jax.numpy as jnp


env = Hopper(render=True)
# q = jnp.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
# qdot = jnp.zeros((6,))
# env.state = jnp.hstack([q, qdot])
action = jnp.zeros((3,))
# action = action.at[0].set(0.1)
for i in range(10000):
    next_state, reward, done, info = env.step(action)
    env.reset_render_state()
    if done:
        print(i, done)
        env.reset(*DEFAULT_PURE_HOPPER_PARAMS)
        # q = jnp.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
        # qdot = jnp.zeros((6,))
        # env.state = jnp.hstack([q, qdot])

