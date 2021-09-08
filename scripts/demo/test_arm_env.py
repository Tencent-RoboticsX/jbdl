from jbdl.envs.arm_env import Arm
import jax.numpy as jnp

env = Arm(render=True)
q = jnp.zeros(7)
qdot = jnp.zeros(7)
state = jnp.array([q, qdot]).flatten()
env.state = state
action = jnp.zeros((7,))


# x = jnp.array([1,2,3])
# y = x[x>2]
# print(y)

for i in range(1000):
    print(i)
    next_state, reward, done, _ = env.step(action)
    env.reset_render_state()