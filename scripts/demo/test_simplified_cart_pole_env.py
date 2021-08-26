from jbdl.envs.simplified_cart_pole_env import SimplifiedCartPole
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


