from jbdl.envs.mountain_car_env import MountainCar
import numpy as np


env = MountainCar(render=True)
for i in range(1000):
    action = np.zeros((1,)) + 1.0
    next_state, reward, done, _ = env.step(action)
    env.reset_render_state()
