from functools import partial
import math
from collections import namedtuple
from jbdl.envs.base_env import BaseEnv
from gym.envs.classic_control import rendering
import jax
import jax.numpy as jnp
from jax.ops import index_update, index
from jax.config import config
config.update("jax_enable_x64", True)

M_CART = 1.0
M_POLE = 0.1
HALF_POLE_LENGTH = 0.5
F_UNIT = 10.0
DEFAULT_PURE_SIMPLIFIED_CART_POLE_PARAMS = (
    M_CART, M_POLE, HALF_POLE_LENGTH, F_UNIT)


class SimplifiedCartPole(BaseEnv):
    def __init__(self, pure_simplifed_cart_pole_params=DEFAULT_PURE_SIMPLIFIED_CART_POLE_PARAMS,
                 reward_fun=None, seed=123, sim_dt=0.02, batch_size=0, render=False,
                 screen_width=600, screen_height=400, render_idx=None):

        self.a_gravity = -9.8

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Render Parameters
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.world_width = self.x_threshold * 2
        self.scale = screen_width / self.world_width
        self.carty = 100.0

        super().__init__(
            pure_simplifed_cart_pole_params, seed=seed, sim_dt=sim_dt, batch_size=batch_size,
            render=render, render_engine=rendering.Viewer(
                screen_width, screen_height),
            render_idx=render_idx)

        def _dynamics_fun_core(pos_vel, action,  m_cart, m_pole, half_pole_length, f_unit, a_gravity):
            _, theta, x_dot, theta_dot = pos_vel
            force = (2.0 * action[0] - 1.0) * f_unit
            costheta = jnp.cos(theta)
            sintheta = jnp.sin(theta)
            polemass_length = m_pole * half_pole_length
            total_mass = m_cart + m_pole

            # For the interested reader:
            # https://coneural.org/florian/papers/05_cart_pole.pdf

            temp = (force + polemass_length * theta_dot **
                    2 * sintheta) / total_mass
            thetaacc = (-a_gravity * sintheta - costheta * temp) / (
                half_pole_length * (4.0 / 3.0 - m_pole * costheta ** 2 / total_mass))
            xacc = temp - polemass_length * thetaacc * costheta / total_mass
            vel_acc = jnp.array([x_dot, theta_dot, xacc, thetaacc])
            return vel_acc

        self._dynamics_fun = partial(
            _dynamics_fun_core, a_gravity=self.a_gravity)

        def _dynamics_step_core(y0, *dynamics_args, sim_dt):
            x, theta, x_dot, theta_dot = y0
            action, m_cart, m_pole, half_pole_length, f_unit = dynamics_args
            pos_vel = jnp.array([x, theta, x_dot, theta_dot])
            vel_acc = self._dynamics_fun(
                pos_vel, action, m_cart, m_pole, half_pole_length, f_unit)
            xacc = vel_acc[2]
            thetaacc = vel_acc[3]

            x = x + sim_dt * x_dot
            x_dot = x_dot + sim_dt * xacc
            theta = theta + sim_dt * theta_dot
            theta_dot = theta_dot + sim_dt * thetaacc
            y_final = jnp.array([x, theta, x_dot, theta_dot])
            return y_final

        self._dynamics_step = partial(_dynamics_step_core, sim_dt=self.sim_dt)

        def _dynamics_step_with_params_core(state, action, *pure_simplified_cart_pole_params):
            m_cart, m_pole, half_pole_length, f_unit = pure_simplified_cart_pole_params
            action = jnp.reshape(jnp.array(action), (-1,))
            dynamics_args = (action, m_cart, m_pole, half_pole_length, f_unit)
            next_state = self._dynamics_step(state, *dynamics_args)
            return next_state

        self._dynamics_step_with_params = _dynamics_step_with_params_core

        self.dynamics_step = jax.jit(self._dynamics_step)
        self.dynamics_fun = jax.jit(self._dynamics_fun)
        self.dynamics_step_with_params = jax.jit(
            self._dynamics_step_with_params)

        # self.dynamics_step = self._dynamics_step
        # self.dynamics_fun = self._dynamics_fun
        # self.dynamics_step_with_params = self._dynamics_step_with_params

        def _done_fun(next_state):
            return False

        self.done_fun = jax.jit(_done_fun)

        def _default_reward_fun(state, action, next_state):
            return -1.0

        if reward_fun is None:
            self._reward_fun = _default_reward_fun
            self.reward_fun = jax.jit(_default_reward_fun)
        else:
            self._reward_fun = reward_fun
            self.reward_fun = jax.jit(reward_fun)

    def _init_pure_params(self, *pure_simplified_cart_pole_params):
        self.m_cart, self.m_pole, self.half_pole_length, self.f_unit = pure_simplified_cart_pole_params

    def _load_render_robot(self, viewer_client):
        polewidth = 10.0
        polelen = self.scale * (2 * self.half_pole_length)
        cartwidth = 50.0
        cartheight = 30.0

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        carttrans = rendering.Transform()
        cart.add_attr(carttrans)
        viewer_client.add_geom(cart)
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        pole.set_color(0.8, 0.6, 0.4)
        poletrans = rendering.Transform(translation=(0, axleoffset))
        pole.add_attr(poletrans)
        pole.add_attr(carttrans)
        viewer_client.add_geom(pole)
        axle = rendering.make_circle(polewidth / 2)
        axle.add_attr(poletrans)
        axle.add_attr(carttrans)
        axle.set_color(0.5, 0.5, 0.8)
        viewer_client.add_geom(axle)
        track = rendering.Line(
            (0, self.carty), (self.screen_width, self.carty))
        track.set_color(0, 0, 0)
        viewer_client.add_geom(track)
        self._pole_geom = pole
        RenderRobot = namedtuple("RenderRobot", "carttrans poletrans")
        render_robot = RenderRobot(carttrans=carttrans, poletrans=poletrans)
        return render_robot

    def _reset_render_state(self, *render_robot_state):
        cartx, carty, polerot = render_robot_state
        self.render_robot.carttrans.set_translation(cartx, carty)
        self.render_robot.poletrans.set_rotation(-polerot)
        self.viewer_client.render(return_rgb_array=False)

    def _get_render_state(self):
        if self.batch_size == 0:
            cart_trans_x = self.state[0]
            pole_trans_rot = self.state[1]
        else:
            cart_trans_x = self.state[self.render_idx, 0]
            pole_trans_rot = self.state[self.render_idx, 1]
        cartx = cart_trans_x * self.scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = self.carty
        polerot = -pole_trans_rot
        return (cartx, carty, polerot)

    def _state_random_initializer(self):
        self.key, subkey = jax.random.split(self.key)
        state = jax.random.uniform(
            subkey, shape=(4,), minval=-0.05, maxval=0.05)
        return state

    def _batch_state_random_initializer(self, idx_list):
        if idx_list is None:
            self.key, subkey = jax.random.split(self.key)
            state = jax.random.uniform(subkey, shape=(
                self.batch_size, 4), minval=-0.05, maxval=0.05)
        else:
            idx_num = len(idx_list)
            self.key, subkey = jax.random.split(self.key)
            update_state = jax.random.uniform(
                subkey, shape=(idx_num, 1), minval=-0.05, maxval=0.05)
            state = index_update(
                self.state,
                index[idx_list, :],
                update_state
            )
        return state

    def _step_fun(self, action):
        action = jnp.array(action)
        dynamics_params = (action, self.m_cart, self.m_pole,
                           self.half_pole_length, self.f_unit)
        next_state = self.dynamics_step(self.state, *dynamics_params)
        done = self.done_fun(next_state)
        reward = self.reward_fun(self.state, action, next_state)
        self.state = next_state
        next_entry = (next_state, reward, done, {})
        return next_entry

    def _batch_step_fun(self, action):
        action = jnp.reshape(jnp.array(action), newshape=(self.batch_size, -1))
        dynamics_params = (action, self.m_cart, self.m_pole,
                           self.half_pole_length, self.f_unit)
        next_state = jax.vmap(self.dynamics_step, (0, 0, None, None, None, None), 0)(
            self.state, *dynamics_params)
        done = jax.vmap(self.done_fun)(next_state)
        reward = jax.vmap(self.reward_fun)(self.state, action, next_state)
        self.state = next_state
        next_entry = (next_state, reward, done, {})
        return next_entry


if __name__ == "__main__":
    env = SimplifiedCartPole(render=True)
    action = jnp.zeros((1,)) + 1.0
    print(env.state)
    for i in range(100):
        next_state, reward, done, _ = env.step(action)
        env.reset_render_state()
        print(next_state, reward, done)
        print(env.state)

    #  Test the consistency with gym env.
    from gym.envs.classic_control import CartPoleEnv
    import numpy as np
    np.set_printoptions(precision=10)
    jnp.set_printoptions(precision=10)
    env = CartPoleEnv()
    jenv = SimplifiedCartPole()
    obs = env.reset()
    jenv.state = jnp.array(obs[[0, 2, 1, 3]])
    state_errors = []
    for i in range(10000):
        action = np.random.randint(0, 2)
        jaction = jnp.array([action, ])
        next_obs, reward, done, _ = env.step(action)
        jnext_obs, jreward, jdone, _ = jenv.step(jaction)
        state_errors.append(next_obs[[0, 2, 1, 3]]-jnext_obs)
        if done:
            obs = env.reset()
            jenv.state = jnp.array(obs[[0, 2, 1, 3]])
    print(np.sum(np.abs(state_errors)))
