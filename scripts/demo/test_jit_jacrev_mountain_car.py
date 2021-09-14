import time
import jax
import jax.numpy as jnp
from jbdl.envs.mountain_car_env import MountainCar

env = MountainCar()

m_car = 0.0025 / 9.81
f_unit = 0.001
pure_env_params = (m_car, f_unit)
state = env.reset(*pure_env_params)
action = jnp.array([1.0, ])


start = time.time()
next_state = env.dynamics_step_with_params(state, action, m_car, f_unit)
print(next_state)
duration = time.time() - start
print("duration:", duration)

start = time.time()
next_state = env.dynamics_step_with_params(state, action, m_car, f_unit)
print(next_state)
duration = time.time() - start
print("duration:", duration)
print("========================")


batch_size = 1000
v_state = jnp.repeat(jnp.expand_dims(state, 0), batch_size, axis=0)
v_action = jnp.repeat(jnp.expand_dims(action, 0), batch_size, axis=0)

start = time.time()
v_dynamics_step_with_params = jax.vmap(env.dynamics_step_with_params, (0, 0, None, None,), 0)
v_next_state = v_dynamics_step_with_params(v_state, v_action, m_car, f_unit)
# print(v_next_state)
duration = time.time() - start
print("duration:", duration)

start = time.time()
v_next_state = v_dynamics_step_with_params(v_state, v_action, m_car, f_unit)
# print(v_next_state)
duration = time.time() - start
print("duration:", duration)
print("==================")


start = time.time()
dns_to_action = jax.jit(jax.jacrev(env.dynamics_step_with_params, argnums=0))
print(dns_to_action(state, action, m_car, f_unit))
duration = time.time() - start
print("duration:", duration)


start = time.time()
print(dns_to_action(state, action, m_car, f_unit))
duration = time.time() - start
print("duration:", duration)

print("==================")


start = time.time()
v_dns_to_action = jax.vmap(dns_to_action, (0, 0, None, None), 0)
print(v_dns_to_action(v_state, v_action, m_car, f_unit))
duration = time.time() - start
print("duration:", duration)


start = time.time()
print(v_dns_to_action(v_state, v_action, m_car, f_unit))
duration = time.time() - start
print("duration:", duration)

