import jax
from jbdl.envs.reacher_env_with_extended_state import ReacherWithExtendedState
import jax.numpy as jnp
import time

env = ReacherWithExtendedState()

m_body0 = 0.5
m_body1 = 2.0
ic_params_body0 = jnp.zeros((6,))
ic_params_body1 = jnp.zeros((6,))
joint_damping_params = jnp.array([0.7, 0.7])

pure_env_params = (m_body0, m_body1, ic_params_body0,
                   ic_params_body1, joint_damping_params)
state = env.reset(*pure_env_params)
action = jnp.array([1.0, 0.0])
extended_state = env.extended_state

print("state: ", state)
print("extended_state: ", extended_state)

print("=====================")

start = time.time()
next_extended_state = env.extended_dynamics_step_with_params(
    env.dynamics_fun, env.extended_state, action, m_body0, m_body1, ic_params_body0, ic_params_body1, joint_damping_params)
print(next_extended_state)
duration = time.time() - start
print("duration:", duration)

start = time.time()
next_extended_state = env.extended_dynamics_step_with_params(
    env.dynamics_fun, env.extended_state, action, m_body0, m_body1, ic_params_body0, ic_params_body1, joint_damping_params)
print(next_extended_state)
duration = time.time() - start
print("duration:", duration)
print("========================")

batch_size = 10
v_extended_state = jnp.repeat(jnp.expand_dims(
    env.extended_state, 0), batch_size, axis=0)
v_action = jnp.repeat(jnp.expand_dims(action, 0), batch_size, axis=0)

start = time.time()
v_extended_dynamics_step_with_params = jax.vmap(
    env.extended_dynamics_step_with_params, (None, 0, 0, None, None, None, None, None), 0)
v_next_extended_state = v_extended_dynamics_step_with_params(
    env.dynamics_fun, v_extended_state, v_action, m_body0, m_body1, ic_params_body0, ic_params_body1, joint_damping_params)
print(v_next_extended_state)
duration = time.time() - start
print("duration:", duration)

start = time.time()
v_next_extended_state = v_extended_dynamics_step_with_params(
    env.dynamics_fun, v_extended_state, v_action, m_body0, m_body1, ic_params_body0, ic_params_body1, joint_damping_params)
print(v_next_extended_state)
duration = time.time() - start
print("duration:", duration)
print("==================")


start = time.time()
dns_to_action = jax.jit(jax.jacrev(
    env.extended_dynamics_step_with_params, argnums=7), static_argnums=0)
print(dns_to_action(env.dynamics_fun, env.extended_state, action, m_body0,
      m_body1, ic_params_body0, ic_params_body1, joint_damping_params))
duration = time.time() - start
print("duration:", duration)


start = time.time()
print(dns_to_action(env.dynamics_fun, env.extended_state, action, m_body0,
      m_body1, ic_params_body0, ic_params_body1, joint_damping_params))
duration = time.time() - start
print("duration:", duration)

print("==================")


start = time.time()
v_dns_to_action = jax.vmap(
    dns_to_action, (None, 0, 0, None, None, None, None, None), 0)
print(v_dns_to_action(env.dynamics_fun, v_extended_state, v_action,
      m_body0, m_body1, ic_params_body0, ic_params_body1, joint_damping_params))
duration = time.time() - start
print("duration:", duration)


start = time.time()
print(v_dns_to_action(env.dynamics_fun, v_extended_state, v_action,
      m_body0, m_body1, ic_params_body0, ic_params_body1, joint_damping_params))
duration = time.time() - start
print("duration:", duration)

print("================")
