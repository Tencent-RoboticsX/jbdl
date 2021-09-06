from jbdl.envs.reacher_env import Reacher
import jax.numpy as jnp


M_BODY0 = 1.0
M_BODY1 = 1.0
IC_PARAMS_BODY0 = jnp.zeros((6,))
IC_PARAMS_BODY1 = jnp.zeros((6,))
JOINT_DAMPING_PARAMS = jnp.array([0.7, 0.7])

PURE_REACHER_PARAMS = (
    M_BODY0, M_BODY1, IC_PARAMS_BODY0, IC_PARAMS_BODY1, JOINT_DAMPING_PARAMS)

env = Reacher(pure_reacher_params=PURE_REACHER_PARAMS, render=True)
action = jnp.array([0.5, -1.0])
print(env.state)
for i in range(1000):
    # print(i)
    next_state, reward, done, _ = env.step(action)
    # print(env.state)
    # env.state = jnp.array([jnp.sin(i/100.0), jnp.cos(i/100.0), 0., 0.])
    env.reset_render_state()

    # if done:
    #     print("done")
    #     env.reset(*DEFAULT_PURE_REACHER_PARAMS)
