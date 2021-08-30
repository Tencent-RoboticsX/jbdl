from jbdl.envs.cart_pole_env import CartPole, DEFAULT_PURE_CART_POLE_PARAMS
import jax.numpy as jnp
import numpy as np

env = CartPole(render=True, batch_size=2)
action = np.zeros((2,1))
print(env.state)
for i in range(10000):
    next_state, reward, done, _ = env.step(action)
    print(next_state)
    env.reset_render_state()

    # if done:
    #     print("done")
    #     env.reset(*DEFAULT_PURE_CART_POLE_PARAMS)
