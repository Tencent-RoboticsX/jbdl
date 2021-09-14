import time
from jbdl.experimental.envs.hopper_env import Hopper
import jax
import jax.numpy as jnp

env = Hopper()

m_torso = 0.4 * 0.05 * 0.05 * 3000
m_thigh = 0.45 * 0.05 * 0.05 * 3000
m_leg = 0.5 * 0.05 * 0.05 * 3000
m_foot = 0.39 * 0.06 * 0.06 * 3000

ic_params_torso = jnp.zeros((6,))
ic_params_thigh = jnp.zeros((6,))
ic_params_leg = jnp.zeros((6,))
ic_params_foot = jnp.zeros((6,))
joint_damping_params = jnp.array([0.7, 0.7, 0.7])

pure_env_params = (
    m_torso, m_thigh, m_leg, m_foot,
    ic_params_torso, ic_params_thigh, ic_params_leg, ic_params_foot, joint_damping_params)

state = env.reset(*pure_env_params)
action = jnp.ones((3,))

start = time.time()
next_state = env.dynamics_step_with_params(
    env.dynamics_fun, env.events_fun, env.impulsive_dynamics_fun, state, action, *pure_env_params)

print(next_state)
duration = time.time() - start
print("duration:", duration)

start = time.time()
next_state = env.dynamics_step_with_params(
     env.dynamics_fun, env.events_fun, env.impulsive_dynamics_fun, state, action, *pure_env_params)
print(next_state)
duration = time.time() - start
print("duration:", duration)
print("========================")


batch_size = 100
v_state = jnp.repeat(jnp.expand_dims(state, 0), batch_size, axis=0)
v_action = jnp.repeat(jnp.expand_dims(action, 0), batch_size, axis=0)

start = time.time()
v_dynamics_step_with_params = jax.vmap(env.dynamics_step_with_params,
    (None, None, None, 0, 0, None, None, None, None, None, None, None, None, None), 0)
v_next_state = v_dynamics_step_with_params(
    env.dynamics_fun, env.events_fun, env.impulsive_dynamics_fun, v_state, v_action, *pure_env_params)
# print(v_next_state)
duration = time.time() - start
print("duration:", duration)

start = time.time()
v_next_state = v_dynamics_step_with_params(
    env.dynamics_fun, env.events_fun, env.impulsive_dynamics_fun, v_state, v_action, *pure_env_params)
# print(v_next_state)
duration = time.time() - start
print("duration:", duration)
print("==================")

start = time.time()
dns_to_action = jax.jit(jax.jacfwd(env.dynamics_step_with_params, argnums=13), static_argnums=[0, 1, 2])
print(dns_to_action(
    env.dynamics_fun, env.events_fun, env.impulsive_dynamics_fun, state, action, *pure_env_params))
duration = time.time() - start
print("duration:", duration)


start = time.time()
print(dns_to_action(
    env.dynamics_fun, env.events_fun, env.impulsive_dynamics_fun, state, action, *pure_env_params))
duration = time.time() - start
print("duration:", duration)

print("==================")

start = time.time()
v_dns_to_action = jax.vmap(dns_to_action,  (None, None, None, 0, 0, None, None, None, None, None, None, None, None, None), 0)
print(v_dns_to_action(
    env.dynamics_fun, env.events_fun, env.impulsive_dynamics_fun, v_state, v_action, *pure_env_params))
duration = time.time() - start
print("duration:", duration)


start = time.time()
print(v_dns_to_action(
    env.dynamics_fun, env.events_fun, env.impulsive_dynamics_fun, v_state, v_action, *pure_env_params))
duration = time.time() - start
print("duration:", duration)

