import math
from functools import partial
import numpy as np
from jbdl.envs.base_env import BaseEnv
from gym.envs.classic_control import mountain_car, rendering
import jax.numpy as jnp
import jax
from jax.ops import index_update, index
from jax.config import config

config.update("jax_enable_x64", True)


M_CAR = 0.0025 / 9.81
F_UNIT = 0.001
DEFAULT_PURE_MOUNTAIN_CAR_PARAMS = (M_CAR, F_UNIT)


class MountainCar(BaseEnv):
    def __init__(self, pure_mountain_car_params=DEFAULT_PURE_MOUNTAIN_CAR_PARAMS,
                 reward_fun=None, seed=123, sim_dt=math.sqrt(0.0025/9.81), batch_size=0,
                 render=False, screen_width=600, screen_height=400, render_idx=None):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_delta_position = 0.07
        self.goal_position = 0.5
        self.a_gravity = -9.81

        super().__init__(pure_mountain_car_params, seed=seed, sim_dt=sim_dt,
                         batch_size=batch_size, render=render,
                         render_engine=rendering.Viewer(screen_width, screen_height),
                         render_idx=render_idx)

        def _dynamics_fun_core(pos_vel, action,  m_car, f_unit, a_gravity):
            position = pos_vel[0:1]
            velocity = pos_vel[1:2]
            acceleration = (action - 1) * f_unit / m_car + \
                jnp.cos(3 * position) * a_gravity
            vel_acc = jnp.hstack([velocity, acceleration])
            return vel_acc

        self._dynamics_fun = partial(
            _dynamics_fun_core, a_gravity=self.a_gravity)

        def _dynamics_step_core(y0, *dynamics_args, min_position, max_position, max_delta_position, sim_dt):
            position = y0[0:1]
            velocity = y0[1:2] / sim_dt
            pos_vel = jnp.hstack([position, velocity])
            action, m_car, f_unit = dynamics_args
            vel_acc = self._dynamics_fun(pos_vel, action, m_car, f_unit)
            velocity += vel_acc[1:2] * sim_dt
            velocity = jnp.clip(velocity, -max_delta_position /
                                sim_dt, max_delta_position/sim_dt)
            position += velocity * sim_dt
            position = jnp.clip(position, min_position, max_position)

            velocity = jax.lax.cond(
                jnp.all(jnp.hstack(
                    [jnp.heaviside(min_position-position, 1.0), jnp.heaviside(-velocity, 0.0)])),
                lambda _: jnp.zeros((1,)),
                lambda _: velocity,
                operand=None)

            y_final = jnp.hstack([position, velocity*sim_dt])
            return y_final

        self._dynamics_step = partial(_dynamics_step_core,
                                      min_position=self.min_position,
                                      max_position=self.max_position,
                                      max_delta_position=self.max_delta_position,
                                      sim_dt=self.sim_dt)

        def _dynamics_step_with_params_core(state, action, *pure_mountain_car_params):
            m_car, f_unit = pure_mountain_car_params
            action = jnp.reshape(jnp.array(action), (-1,))
            dynamics_args = (action, m_car, f_unit)
            next_state = self._dynamics_step(state, *dynamics_args)
            return next_state

        self._dynamics_step_with_params = _dynamics_step_with_params_core

        self.dynamics_step = jax.jit(self._dynamics_step)
        self.dynamics_fun = jax.jit(self._dynamics_fun)
        self.dynamics_step_with_params = jax.jit(
            self._dynamics_step_with_params)

        def _done_fun(next_state):
            return jnp.all(jnp.hstack([jnp.heaviside(next_state[0:1]-0.5, 1.0), jnp.heaviside(next_state[1:2], 1.0)]))

        self.done_fun = jax.jit(_done_fun)

        def _default_reward_fun(state, action, next_state):
            return -1.0

        if reward_fun is None:
            self._reward_fun = _default_reward_fun
            self.reward_fun = jax.jit(_default_reward_fun)
        else:
            self._reward_fun = reward_fun
            self.reward_fun = jax.jit(reward_fun)

    def _init_pure_params(self, *pure_mountain_car_params):
        self.m_car, self.f_unit = pure_mountain_car_params

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def _load_render_robot(self, viewer_client):
        screen_width = viewer_client.width
        # screen_height = viewer_client.height
        world_width = self.max_position - self.min_position
        self.scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * self.scale, ys * self.scale))
        track = rendering.make_polyline(xys)
        track.set_linewidth(4)
        viewer_client.add_geom(track)
        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        car.add_attr(rendering.Transform(translation=(0, clearance)))
        cartrans = rendering.Transform()
        car.add_attr(cartrans)
        viewer_client.add_geom(car)

        frontwheel = rendering.make_circle(carheight / 2.5)
        frontwheel.set_color(0.5, 0.5, 0.5)
        frontwheel.add_attr(
            rendering.Transform(translation=(carwidth / 4, clearance))
        )
        frontwheel.add_attr(cartrans)
        viewer_client.add_geom(frontwheel)

        backwheel = rendering.make_circle(carheight / 2.5)
        backwheel.add_attr(
            rendering.Transform(translation=(-carwidth / 4, clearance))
        )
        backwheel.add_attr(cartrans)
        backwheel.set_color(0.5, 0.5, 0.5)
        viewer_client.add_geom(backwheel)

        flagx = (self.goal_position - self.min_position) * self.scale
        flagy1 = self._height(self.goal_position) * self.scale
        flagy2 = flagy1 + 50
        flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
        viewer_client.add_geom(flagpole)
        flag = rendering.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )
        flag.set_color(0.8, 0.8, 0)
        viewer_client.add_geom(flag)
        return cartrans

    def _reset_render_state(self, *render_robot_state):
        car_trans_x, car_trans_y, car_trans_rot = render_robot_state
        self.render_robot.set_translation(car_trans_x, car_trans_y)
        self.render_robot.set_rotation(car_trans_rot)
        self.viewer_client.render(return_rgb_array=False)

    def _get_render_state(self):
        if self.batch_size == 0:
            pos = self.state[0]
        else:
            pos = self.state[self.render_idx, 0]
        car_trans_x = (pos - self.min_position) * self.scale
        car_trans_y = self._height(pos) * self.scale
        car_trans_rot = math.cos(3 * pos)
        return (car_trans_x, car_trans_y, car_trans_rot)

    def _state_random_initializer(self):
        self.key, subkey = jax.random.split(self.key)
        state = np.array(
            [jax.random.uniform(subkey, minval=-0.6, maxval=-0.4), 0.0])
        return state

    def _batch_state_random_initializer(self, idx_list):
        if idx_list is None:
            self.key, subkey = jax.random.split(self.key)
            state = jnp.concatenate([jax.random.uniform(subkey, shape=(
                self.batch_size, 1), minval=-0.6, maxval=-0.4), jnp.zeros((self.batch_size, 1))], axis=1)
            return state
        else:
            idx_num = len(idx_list)
            self.key, subkey = jax.random.split(self.key)
            update_state = jnp.concatenate([jax.random.uniform(subkey, shape=(
                idx_num, 1), minval=-0.6, maxval=-0.4), jnp.zeros((idx_num, 1))], axis=1)
            state = index_update(
                self.state,
                index[idx_list, :],
                update_state
            )
            return state

    def _step_fun(self, action):
        action = jnp.array(action)
        dynamics_params = (action, self.m_car, self.f_unit)
        next_state = self.dynamics_step(self.state, *dynamics_params)
        done = self.done_fun(next_state)
        reward = self.reward_fun(self.state, action, next_state)
        self.state = next_state
        next_entry = (next_state, reward, done, {})
        return next_entry

    def _batch_step_fun(self, action):
        action = jnp.reshape(jnp.array(action), newshape=(self.batch_size, -1))
        dynamics_params = (action, self.m_car, self.f_unit)
        next_state = jax.vmap(self.dynamics_step, (0, 0, None, None), 0)(
            self.state, *dynamics_params)
        done = jax.vmap(self.done_fun)(next_state)
        reward = jax.vmap(self.reward_fun)(self.state, action, next_state)
        self.state = next_state
        next_entry = (next_state, reward, done, {})
        return next_entry


if __name__ == "__main__":
    # mountain_car = MountainCar(batch_size=2)
    # action =  jnp.zeros((2, 1))
    mountain_car = MountainCar()
    action = jnp.zeros((1,))
    print(mountain_car.state)
    next_state, reward, done, _ = mountain_car.step(action)
    print(next_state, reward, done)
    print(mountain_car.state)

    # Test the consistency with gym env.
    # from gym.envs.classic_control import MountainCarEnv
    # np.set_printoptions(precision=10)
    # jnp.set_printoptions(precision=10)
    # env = MountainCarEnv()
    # jenv = MountainCar()
    # obs = env.reset()
    # jenv.state = jnp.array(obs)
    # # print(obs, jenv.state)
    # # print(jenv.state.dtype)
    # state_errors = []
    # for i in range(10000):
    #     action = np.random.randint(0, 3)
    #     jaction = jnp.array([action,])
    #     # print("action:", action)
    #     next_obs, reward, done, _ = env.step(action)
    #     jnext_obs, jreward, jdone, _ = jenv.step(jaction)
    #     state_errors.append(next_obs-jnext_obs)
    #     if done:
    #         obs = env.reset()
    #         jenv.state = jnp.array(obs)

    #     # print(next_obs, jnext_obs)
    #     # print(jnext_obs.dtype)
    # print(np.sum(np.abs(state_errors)))
