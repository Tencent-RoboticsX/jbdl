from abc import ABC, abstractmethod

from numpy.core.fromnumeric import shape
from jbdl.rbdl.model.rigid_body_inertia import rigid_body_inertia, init_ic_by_cholesky
import pybullet
import jax
import jax.numpy as jnp


class BaseEnv(ABC):
    def __init__(self, pure_env_params, seed=0,
                 sim_dt=0.1, rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf, batch_size=0,
                 render=False, render_engine=pybullet, render_idx=None):

        self.sim_dt = sim_dt
        self.rtol = rtol
        self.atol = atol
        self.mxstep = mxstep
        self.batch_size = batch_size


        self.key = jax.random.PRNGKey(seed)
        self.reset(*pure_env_params, idx_list=None)


        self.render = render
        if render_idx is None:
            self.render_idx = 0
        else:
            self.render_idx = render_idx

        if self.render:
            self.viewer_client = render_engine
            self.render_robot = self._load_render_robot(self.viewer_client)

    @abstractmethod
    def _init_pure_params(self, *pure_env_params):
        '''Implement in subclass'''

    @abstractmethod
    def _load_render_robot(self, viewer_client):
        '''Implement in subclass'''

    @abstractmethod
    def _reset_render_state(self, *render_robot_state):
        '''Implement in subclass'''

    @abstractmethod
    def _get_render_state(self):
        '''Implement in subclass'''

    @property
    def render_state(self):
        return self._get_render_state()

    def reset_render_state(self):
        if self.render:
            self._reset_render_state(*self.render_state)

    @abstractmethod
    def _state_random_initializer(self):
        '''Implement in subclass'''

    @abstractmethod
    def _batch_state_random_initializer(self, idx_list):
        '''Implement in subclass'''

    @abstractmethod
    def _step_fun(self, action):
        '''Implement in subclass'''

    @abstractmethod
    def _batch_step_fun(self, action):
        '''Implement in subclass'''

    def reset(self, *pure_env_params, **kwargs):
        self._init_pure_params(*pure_env_params)
        idx_list = kwargs.get("idx_list", None)
        if self.batch_size == 0:
            self.state = self._state_random_initializer()
        else:
            self.state = self._batch_state_random_initializer(idx_list)
        return self.state

    def step(self, action):
        if self.batch_size == 0:
            next_entry = self._step_fun(action)
        else:
            next_entry = self._batch_step_fun(action)
        return next_entry

    # def draw_line(self, from_pos, to_pos, line_color_rgb=[0, 1, 0], line_width=2):
    #     pass
    def _action_wrapper(self, action):
        update_action = jnp.reshape(jnp.array(action), newshape=(-1,))
        return update_action

    def _batch_action_wrapper(self, action):
        update_action = jnp.reshape(jnp.array(action), newshape=(self.batch_size, -1))
        return update_action

    @staticmethod
    def init_inertia(m, c, l):
        ic = init_ic_by_cholesky(l)
        inertia = rigid_body_inertia(m, c, ic)
        return inertia

    # @staticmethod
    # @abstractmethod
    # def _dynamics_fun(y, t, *pargs, **kwargs):
    #     '''Implement in subclass'''

    # @staticmethod
    # @abstractmethod
    # def _dynamics_step(dynamics_fun, state, action, *pargs, **kwargs):
    #     '''Implement in subclass'''

    # @staticmethod
    # @abstractmethod
    # def _dynamics_step_with_parameters(dynamics_fun, state, action, *pargs, **kwargs):
    #     '''Implement in subclass'''

    # @staticmethod
    # @abstractmethod
    # def _done_fun(state, *args, **kwargs):
    #     '''Implement in subclass'''

    # @staticmethod
    # @abstractmethod
    # def _default_rwd_fun(state, action, next_state, *pargs, **kwargs):
    #     '''Implement in subclass'''
