import jax
import jax.numpy as jnp
from jbdl.envs.reacher_env import Reacher, DEFAULT_PURE_REACHER_PARAMS


class ReacherWithExtendedState(Reacher):
    def __init__(
            self, pure_reacher_params=DEFAULT_PURE_REACHER_PARAMS, reward_fun=None,
            seed=3, sim_dt=0.1, rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf, batch_size=0,
            render=False, render_engine_name="pybullet", render_idx=None):

        def _state_wrapper(state):
            q = state[0:2]
            sin_cos_q = jnp.hstack([jnp.sin(q), jnp.cos(q)])
            extended_state = jnp.hstack([sin_cos_q, state[2:]])
            return extended_state

        def _state_unwrapper(extended_state):
            sin_q = extended_state[0:2]
            cos_q = extended_state[2:4]
            q = jnp.arctan(sin_q/cos_q)
            state = jnp.hstack([q, extended_state[4:]])
            return state

        self._state_wrapper = _state_wrapper
        self._state_unwrapper = _state_unwrapper
        self.state_wrapper = jax.jit(self._state_wrapper)
        self.state_unwrapper = jax.jit(self._state_unwrapper)

        super().__init__(
            pure_reacher_params=pure_reacher_params, reward_fun=reward_fun,
            seed=seed, sim_dt=sim_dt, rtol=rtol, atol=atol, mxstep=mxstep, batch_size=batch_size,
            render=render, render_engine_name=render_engine_name, render_idx=render_idx)

        def _extended_dynamics_step_with_params(dynamics_fun, extended_state, action, *pure_reacher_params):
            state = self._state_unwrapper(extended_state)
            next_state = self._dynamics_step_with_params(
                dynamics_fun, state, action, *pure_reacher_params)
            next_extended_state = self._state_wrapper(next_state)
            return next_extended_state

        self._extended_dynamics_step_with_params = _extended_dynamics_step_with_params
        self.extended_dynamics_step_with_params = jax.jit(
            self._extended_dynamics_step_with_params, static_argnums=0)

    @property
    def extended_state(self):
        if self.batch_size == 0:
            return self.state_wrapper(self.state)
        elif self.batch_size > 0:
            return jax.vmap(self.state_wrapper)(self.state)
        else:
            raise NotImplementedError()


if __name__ == "__main__":
    env = ReacherWithExtendedState(batch_size=2)
    print(env.extended_state)
