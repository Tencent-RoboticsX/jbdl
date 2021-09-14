from functools import partial
import pybullet
import jax
from jax.ops import index, index_update
import jax.numpy as jnp
from jbdl.envs.base_env import BaseEnv
from jbdl.envs.utils.parser import MJCFBasedRobot
from jbdl.rbdl.utils import xyz2int
from jbdl.rbdl.math import x_trans
from jbdl.rbdl.kinematics import calc_body_to_base_coordinates_core
from jbdl.rbdl.dynamics import forward_dynamics_core
from jbdl.experimental.contact.calc_joint_damping import calc_joint_damping_core
from jbdl.experimental.ode.runge_kutta import odeint

M_CART = 0.1 * 0.1 * 0.2 * 3000
M_POLE = 0.045 * 0.045 * 0.6 * 3000
M_POLE2 = 0.045 * 0.045 * 0.6 * 3000
POLE_IC_PARAMS = jnp.zeros((6,))
POLE2_IC_PARAMS = jnp.zeros((6,))
HALF_POLE_LENGTH = 0.3
HALF_POLE2_LENGTH = 0.3
JOINT_DAMPING_PARAMS = jnp.array([0.7, 0.7, 0.7])

DEFAULT_PURE_INVERTED_DOUBLE_PENDULUM_PARAMS = (
    M_CART, M_POLE, M_POLE2,
    POLE_IC_PARAMS, POLE2_IC_PARAMS,
    HALF_POLE_LENGTH, HALF_POLE2_LENGTH, JOINT_DAMPING_PARAMS)


class InvertedDoublePendulum(BaseEnv):
    def __init__(
            self, pure_inverted_double_pendulum_params=DEFAULT_PURE_INVERTED_DOUBLE_PENDULUM_PARAMS,
            reward_fun=None, seed=123, sim_dt=0.1, rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf, batch_size=0,
            render=False, render_engine_name="pybullet", render_idx=None):

        self.nb = 3
        self.nf = 3
        self.a_grav = jnp.array([[0.], [0.], [0.], [0.], [0.], [-9.81]])
        self.jtype = (1, 0, 0)
        self.jaxis = xyz2int('xyy')
        self.parent = (0, 1, 2)
        self.bid_fingertip = 3
        self.render_engine_name = render_engine_name

        if render_engine_name == "pybullet":
            render_engine = pybullet
        elif render_engine_name == "xmirror":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        def _calc_gpos_fingertip_core(
                q, point_pos, x_tree, parent, jtype, jaxis, body_id):
            pos_fingertip = calc_body_to_base_coordinates_core(
                x_tree, parent, jtype, jaxis, body_id, q, point_pos)
            return jnp.reshape(pos_fingertip, (-1,))

        self._calc_gpos_fingertip = partial(_calc_gpos_fingertip_core,
                                            parent=self.parent, jtype=self.jtype, jaxis=self.jaxis,
                                            body_id=self.bid_fingertip)

        self.calc_gpos_fingertip = jax.jit(self._calc_gpos_fingertip)

        super().__init__(
            pure_inverted_double_pendulum_params, seed=seed,
            sim_dt=sim_dt, rtol=rtol, atol=atol, mxstep=mxstep, batch_size=batch_size,
            render=render, render_engine=render_engine, render_idx=render_idx)

        def _dynamics_fun_core(y, t, x_tree, inertia, joint_damping_params, u, a_grav, parent, jtype, jaxis, nb):
            q = y[0:nb]
            qdot = y[nb:2 * nb]
            joint_damping_tau = calc_joint_damping_core(
                qdot, joint_damping_params)
            tau = u + joint_damping_tau
            input = (x_tree, inertia, parent, jtype,
                     jaxis, nb, q, qdot, tau, a_grav)
            qddot = forward_dynamics_core(*input)
            ydot = jnp.hstack([qdot, qddot])
            return ydot

        self._dynamics_fun = partial(
            _dynamics_fun_core, parent=self.parent, jtype=self.jtype, jaxis=self.jaxis, nb=self.nb)

        def _dynamics_step_core(dynamics_fun, y0, pos_fingertip, *args, nb, sim_dt, rtol, atol, mxstep):
            y_init = y0[0:2 * nb]
            x_tree = args[0]
            t_eval = jnp.linspace(0, sim_dt, 2)
            y_all = odeint(dynamics_fun, y_init, t_eval, *args,
                           rtol=rtol, atol=atol, mxstep=mxstep)
            y_terminal = y_all[-1, :]
            gpos_fingertip = self._calc_gpos_fingertip(
                y_terminal[0:nb], pos_fingertip, x_tree)
            gpos_fingertip = gpos_fingertip[jnp.array([0, 2])]
            y_final = jnp.hstack([y_terminal, gpos_fingertip])
            return y_final

        self._dynamics_step = partial(
            _dynamics_step_core, nb=self.nb, sim_dt=self.sim_dt, rtol=self.rtol, atol=self.atol, mxstep=self.mxstep)

        def _dynamics_step_with_params_core(
                dynamics_fun, state, action, *pure_inverted_double_pendulum_params, a_grav):

            m_cart, m_pole, m_pole2,\
                pole_ic_params, pole2_ic_params,\
                half_pole_length, half_pole2_length, \
                joint_damping_params = pure_inverted_double_pendulum_params

            x_tree = [
                jnp.eye(6),
                jnp.eye(6),
                x_trans(jnp.array([0.0, 0.0, 2 * half_pole_length]))
            ]

            inertia = [
                self.init_inertia(m_cart, jnp.zeros((3,)), jnp.zeros((6,))),
                self.init_inertia(m_pole, jnp.array(
                    [0.0, 0.0, half_pole_length]), pole_ic_params),
                self.init_inertia(m_pole2, jnp.array(
                    [0.0, 0.0, half_pole2_length]), pole2_ic_params)]

            pos_fingertip = jnp.array([0.0, 0.0, 2.0 * self.half_pole2_length])
            
            u = self._action_wrapper(action)

            dynamics_fun_param = (
                x_tree, inertia, joint_damping_params, u, a_grav)
            next_state = self._dynamics_step(
                dynamics_fun, state, pos_fingertip, *dynamics_fun_param)
            return next_state

        self._dynamics_step_with_params = partial(
            _dynamics_step_with_params_core, a_grav=self.a_grav)

        self.dynamics_fun = jax.jit(self._dynamics_fun)
        self.dynamics_step = jax.jit(self._dynamics_step, static_argnums=0)
        self.dynamics_step_with_params = jax.jit(
            self._dynamics_step_with_params, static_argnums=0)

        def _done_fun(next_state):
            posz_fingertip = next_state[7]
            done = jnp.heaviside(1.0 - posz_fingertip, 1.0)
            return done

        self.done_fun = jax.jit(_done_fun)

        def _default_reward_fun(state, action, next_state):
            posx_fingertip = next_state[6]
            posz_fingertip = next_state[7]
            dist_penalty = 0.01 * posx_fingertip ** 2 + \
                (posz_fingertip - 2) ** 2
            alive_bonus = 10
            reward = alive_bonus + dist_penalty
            return reward

        if reward_fun is None:
            self._reward_fun = _default_reward_fun
            self.reward_fun = jax.jit(_default_reward_fun)
        else:
            self._reward_fun = reward_fun
            self.reward_fun = jax.jit(reward_fun)

    def _init_pure_params(self, *pure_env_params):
        self.m_cart, self.m_pole, self.m_pole2, \
            self.pole_ic_params, self.pole2_ic_params, \
            self.half_pole_length, self.half_pole2_length, \
            self.joint_damping_params = pure_env_params
        self.x_tree = [
            jnp.eye(6),
            jnp.eye(6),
            x_trans(jnp.array([0.0, 0.0, 2 * self.half_pole_length]))
        ]
        self.inertia = [
            self.init_inertia(self.m_cart, jnp.zeros((3,)), jnp.zeros((6,))),
            self.init_inertia(self.m_pole, jnp.array(
                [0.0, 0.0, self.half_pole_length]), self.pole_ic_params),
            self.init_inertia(self.m_pole2, jnp.array(
                [0.0, 0.0, self.half_pole2_length]), self.pole2_ic_params)
        ]

        self.pos_fingertip = jnp.array(
            [0.0, 0.0, 2.0 * self.half_pole2_length])

    def _load_render_robot(self, viewer_client):
        render_robot = None
        if self.render_engine_name == "pybullet":
            viewer_client.connect(viewer_client.GUI)
            viewer_client.resetDebugVisualizerCamera(
                cameraDistance=3.0,  cameraYaw=0.0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])
            render_robot = MJCFBasedRobot(
                "inverted_double_pendulum.xml", "cart", action_dim=1, obs_dim=9)
            render_robot.load(viewer_client)
        elif self.render_engine_name == "xmirror":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return render_robot

    def _get_render_state(self):
        if self.batch_size == 0:
            return (self.state[0], self.state[1], self.state[2])
        else:
            return (self.state[self.render_idx, 0], self.state[self.render_idx, 1], self.state[self.render_idx, 2])

    def _reset_render_state(self, *render_robot_state):
        x, theta, gamma = render_robot_state
        self.render_robot.jdict["slider"].reset_current_position(x, 0)
        self.render_robot.jdict["hinge"].reset_current_position(theta, 0)
        self.render_robot.jdict["hinge2"].reset_current_position(gamma, 0)

    def _state_random_initializer(self):
        self.key, subkey = jax.random.split(self.key)
        q = jax.random.uniform(
            subkey, shape=(3,), minval=-0.1, maxval=0.1)
        qdot = jnp.zeros((3,))
        gpos_fingertip = self._calc_gpos_fingertip(
            q, self.pos_fingertip, self.x_tree)
        gpos_fingertip = gpos_fingertip[jnp.array([0, 2])]
        state = jnp.hstack([q, qdot, gpos_fingertip])
        return state

    def _batch_state_random_initializer(self, idx_list):
        if idx_list is None:
            self.key, subkey = jax.random.split(self.key)
            batch_q = jax.random.uniform(subkey, shape=(
                self.batch_size, 3), minval=-0.1, maxval=0.1)
            batch_qdot = jnp.zeros((self.batch_size, 3))
            batch_gpos_fingertip = jax.vmap(self.calc_gpos_fingertip, (0, None, None))(
                batch_q, self.pos_fingertip, self.x_tree)
            batch_gpos_fingertip = batch_gpos_fingertip[:, jnp.array([0, 2])]
            state = jnp.hstack([batch_q, batch_qdot, batch_gpos_fingertip])
        else:
            idx_num = len(idx_list)
            self.key, subkey = jax.random.split(self.key)
            update_batch_q = jax.random.uniform(
                subkey, shape=(idx_num, 3), minval=-0.1, maxval=0.1)
            update_batch_qdot = jnp.zeros((idx_num, 3))
            update_batch_gpos_fingertip = jax.vmap(
                self.calc_gpos_fingertip, (0, None, None))(update_batch_q, self.pos_fingertip, self.x_tree)
            update_batch_gpos_fingertip = update_batch_gpos_fingertip[:, jnp.array([
                                                                                   0, 2])]
            update_state = jnp.hstack(
                [update_batch_q, update_batch_qdot, update_batch_gpos_fingertip])
            state = index_update(
                self.state,
                index[idx_list, :],
                update_state
            )
        return state

    def _action_wrapper(self, action):
        action = super()._action_wrapper(action)
        update_action = jnp.hstack([200.0 * jnp.clip(action, -1, 1), jnp.zeros((2,))])
        return update_action

    def _batch_action_wrapper(self, action):
        action = super()._batch_action_wrapper(action)
        update_action = jnp.hstack([200.0 * jnp.clip(action, -1, 1), jnp.zeros((self.batch_size, 2))])
        return update_action

    def _step_fun(self, action):
        u = self._action_wrapper(action)
        dynamics_params = (self.x_tree, self.inertia,
                           self.joint_damping_params, u, self.a_grav)
        next_state = self.dynamics_step(
            self.dynamics_fun, self.state, self.pos_fingertip, *dynamics_params)
        done = self.done_fun(next_state)
        reward = self.reward_fun(self.state, action, next_state)
        self.state = next_state
        next_entry = (next_state, reward, done, {})
        return next_entry

    def _batch_step_fun(self, action):
        u = self._batch_action_wrapper(action)
        dynamics_params = (self.x_tree, self.inertia,
                           self.joint_damping_params, u, self.a_grav)
        next_state = jax.vmap(self.dynamics_step, (None, 0, None, None, None, None, 0, None), 0)(
            self.dynamics_fun, self.state, self.pos_fingertip, *dynamics_params)
        done = jax.vmap(self.done_fun)(next_state)
        reward = jax.vmap(self.reward_fun)(self.state, action, next_state)
        self.state = next_state
        next_entry = (next_state, reward, done, {})
        return next_entry

    # def draw_line(self, from_pos, to_pos, line_color_rgb, line_width):
    #     if self.render:
    #         self.viewer_client.addUserDebugLine(from_pos, to_pos, line_color_rgb, line_width)


if __name__ == "__main__":
    env = InvertedDoublePendulum(render=True)
    # print(env.state)
    # action = jnp.zeros((1,))
    # env._step_fun(action)

    for i in range(1000):
        env.state = jnp.array([0., 0., jnp.sin(i / 100.0), 0., 0., 0.])
        env.reset_render_state()
    #     pos_tip = env.calc_gpos_fingertip(env.state[0:env.nb], env.pos_fingertip, env.x_tree)
    #     env.draw_line([0.0, 0.0, 0.0], pos_tip, "green", 4.0)

    # HALF_POLE2_LENGTH = 0.4
    # PURE_INVERTED_DOUBLE_PENDULUM_PARAMS = (
    #     M_CART, M_POLE, M_POLE2,
    #     POLE_IC_PARAMS, POLE2_IC_PARAMS,
    #     HALF_POLE_LENGTH, HALF_POLE2_LENGTH)

    # env.reset(*PURE_INVERTED_DOUBLE_PENDULUM_PARAMS)

    # for i in range(500):
    #     env.state = jnp.array([0., 0., jnp.sin(i / 100.0), 0., 0., 0.])

    #     env.reset_render_state()
    #     pos_tip = env.calc_gpos_fingertip(env.state[0:env.nb], env.pos_fingertip, env.x_tree)
    #     env.draw_line([0.0, 0.0, 0.0], pos_tip, "green", 4.0)
