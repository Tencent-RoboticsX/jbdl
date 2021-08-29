from functools import partial
import jax
import jax.numpy as jnp
from jax.ops import index_update, index
from jbdl.rbdl.dynamics.forward_dynamics import forward_dynamics_core
from jbdl.rbdl.utils import xyz2int
from jbdl.rbdl.math import x_trans
from jbdl.rbdl.kinematics import calc_body_to_base_coordinates_core
from jbdl.envs.utils.parser import MJCFBasedRobot
from jbdl.envs.base_env import BaseEnv
from jbdl.experimental.ode.runge_kutta import odeint


M_BODY0 = 1.0
M_BODY1 = 1.0
IC_PARAMS_BODY0 = jnp.zeros((6,))
IC_PARAMS_BODY1 = jnp.zeros((6,))

DEFAULT_PURE_REACHER_PARAMS = (
    M_BODY0, M_BODY1, IC_PARAMS_BODY0, IC_PARAMS_BODY1)


class Reacher(BaseEnv):
    def __init__(self, pure_reacher_params=DEFAULT_PURE_REACHER_PARAMS, reward_fun=None,
                 seed=3, sim_dt=0.1, rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf, batch_size=0,
                 render=False, render_idx=None):

        self.nb = 2
        self.nf = 3
        self.a_grav = jnp.array([[0.], [0.], [0.], [0.], [0.], [-9.81]])
        self.jtype = (0, 0)
        self.jaxis = xyz2int('zz')
        self.parent = (0, 1)
        self.x_tree = [jnp.eye(6), x_trans(jnp.array([0.1, 0.0, 0.0]))]

        self.bid_fingertip = 2
        self.pos_fingertip = jnp.array([0.1, 0.0, 0.0])
        self.orig_target = jnp.array([0.1, -0.1, 0.0])

        def _calc_pos_fingertip_core(
                q, point_pos, x_tree, parent, jtype, jaxis, body_id):
            pos_fingertip = calc_body_to_base_coordinates_core(
                x_tree, parent, jtype, jaxis, body_id, q, point_pos)
            return jnp.reshape(pos_fingertip, (-1,))

        self._calc_pos_fingertip = partial(_calc_pos_fingertip_core, point_pos=self.pos_fingertip, x_tree=self.x_tree,
                                           parent=self.parent, jtype=self.jtype, jaxis=self.jaxis, body_id=self.bid_fingertip)

        self.calc_pos_fingertip = jax.jit(self._calc_pos_fingertip)

        def _calc_pos_target_core(target, orig_target=self.orig_target):
            pos_target = jnp.concatenate([target, jnp.zeros(1,)]) + orig_target
            return pos_target

        self._calc_pos_target = partial(
            _calc_pos_target_core, orig_target=self.orig_target)
        self.calc_pos_target = jax.jit(self._calc_pos_target)

        def _calc_potential(to_target_vec):
            return -100 * jnp.linalg.norm(to_target_vec)

        self.calc_potential = _calc_potential

        super().__init__(pure_reacher_params, seed=seed,
                         sim_dt=sim_dt, rtol=rtol, atol=atol, mxstep=mxstep,
                         batch_size=batch_size, render=render, render_idx=render_idx)

        def _dynamics_fun_core(y, t, x_tree, inertia, u, a_grav, parent, jtype, jaxis, nb):
            q = y[0:nb]
            qdot = y[nb:2*nb]
            input = (x_tree, inertia, parent, jtype,
                     jaxis, nb, q, qdot, u, a_grav)
            qddot = forward_dynamics_core(*input)
            ydot = jnp.hstack([qdot, qddot])
            return ydot

        self._dynamics_fun = partial(
            _dynamics_fun_core, parent=self.parent, jtype=self.jtype, jaxis=self.jaxis, nb=self.nb)

        def _dynamics_step_core(dynamics_fun, y0, *args, nb, sim_dt, rtol, atol, mxstep):
            y_init = y0[0:2*nb]
            target = y0[2*nb:2*nb+2]
            t_eval = jnp.linspace(0, sim_dt, 2)

            y_all = odeint(dynamics_fun, y_init, t_eval, *args,
                           rtol=rtol, atol=atol, mxstep=mxstep)
            y_terminal = y_all[-1, :]

            to_target_vec = self.calc_pos_target(
                target) - self.calc_pos_fingertip(y_terminal[0:nb])
            y_final = jnp.concatenate([y_terminal, target, to_target_vec[0:2]])
            return y_final

        self._dynamics_step = partial(
            _dynamics_step_core, nb=self.nb, sim_dt=self.sim_dt, rtol=self.rtol, atol=self.atol, mxstep=self.mxstep)

        def _dynamics_step_with_params_core(dynamics_fun, state, action, *pure_reacher_params, x_tree, a_grav):
            m_body0, m_body1, ic_params_body0, ic_params_body1 = pure_reacher_params
            inertia_body0 = self.init_inertia(m_body0, jnp.array(
                [0.05, 0.0, 0.0]), ic_params_body0)
            inertia_body1 = self.init_inertia(m_body1, jnp.array(
                [0.05, 0.0, 0.0]), ic_params_body1)
            inertia = [inertia_body0, inertia_body1]
            u = jnp.reshape(jnp.array(action), (-1,))
            dynamics_fun_param = (x_tree, inertia, u, a_grav)
            next_state = self._dynamics_step(
                dynamics_fun, state, *dynamics_fun_param)
            return next_state

        self._dynamics_step_with_params = partial(
            _dynamics_step_with_params_core, x_tree=self.x_tree, a_grav=self.a_grav)

        self.dynamics_fun = jax.jit(self._dynamics_fun)
        self.dynamics_step = jax.jit(self._dynamics_step, static_argnums=0)
        self.dynamics_step_with_params = jax.jit(
            self._dynamics_step_with_params, static_argnums=0)

        def _done_fun(next_state):
            return False

        self.done_fun = jax.jit(_done_fun)

        def _default_reward_fun(state, action, next_state):
            potential_old = self.calc_potential(state[-2:])
            potential = self.calc_potential(next_state[-2:])
            delta_potential = potential - potential_old
            # theta = next_state[0]
            gamma = next_state[1]
            theta_dot = next_state[0]
            gamma_dot = next_state[1]

            # work torque*angular_velocity plus  stall torque require some energy
            electricity_cost = -0.10 * (jnp.abs(action[0] * theta_dot) + jnp.abs(
                action[1] * gamma_dot)) - 0.01 * (jnp.abs(action[0]) + jnp.abs(action[1]))
            stuck_joint_cost = jnp.where(
                jnp.abs(jnp.abs(gamma) - 1) < 0.01, -1.0, 0.0)

            reward = delta_potential + electricity_cost + stuck_joint_cost
            return reward

        if reward_fun is None:
            self._reward_fun = _default_reward_fun
            self.reward_fun = jax.jit(_default_reward_fun)
        else:
            self._reward_fun = reward_fun
            self.reward_fun = jax.jit(reward_fun)

    def _init_pure_params(self, *pure_params):
        self.m_body0, self.m_body1, self.ic_params_body0, self.ic_params_body1 = pure_params
        self.inertia_body0 = self.init_inertia(self.m_body0, jnp.array(
            [0.05, 0.0, 0.0]), self.ic_params_body0)
        self.inertia_body1 = self.init_inertia(self.m_body1, jnp.array(
            [0.05, 0.0, 0.0]), self.ic_params_body1)
        self.inertia = [self.inertia_body0, self.inertia_body1]

    def _load_render_robot(self, viewer_client):
        viewer_client.connect(viewer_client.GUI)
        viewer_client.resetDebugVisualizerCamera(
            cameraDistance=2.0,  cameraYaw=-90, cameraPitch=-60, cameraTargetPosition=[0, 0, 0])
        render_robot = MJCFBasedRobot(
            "reacher.xml", "body0", action_dim=2, obs_dim=8)
        render_robot.load(viewer_client)
        return render_robot

    def _reset_render_state(self, *render_state):
        central_joint, elbow_joint, target_x, target_y, = render_state
        self.render_robot.jdict["target_x"].reset_current_position(target_x, 0)
        self.render_robot.jdict["target_y"].reset_current_position(target_y, 0)
        self.render_robot.jdict["joint0"].reset_current_position(
            central_joint, 0)
        self.render_robot.jdict["joint1"].reset_current_position(
            elbow_joint, 0)

    def _get_render_state(self):
        if self.batch_size == 0:
            return (self.state[0], self.state[1], self.state[4], self.state[5])
        else:
            return (
                self.state[self.render_idx, 0],
                self.state[self.render_idx, 1],
                self.state[self.render_idx, 4],
                self.state[self.render_idx, 5])

    def _state_random_initializer(self):
        self.key, subkey = jax.random.split(self.key)
        q = jax.random.uniform(subkey, shape=(2,), minval=-3.14, maxval=3.14)
        qdot = jnp.zeros((2,))
        self.key, subkey = jax.random.split(self.key)
        target = jax.random.uniform(subkey, (2,), minval=-0.27, maxval=0.27)
        to_target_vec = self.calc_pos_target(
            target) - self.calc_pos_fingertip(q)
        state = jnp.concatenate([q, qdot, target, to_target_vec[0:2]])
        # q(2): joint angle, qdot(2): joint angle velocity, target: x, y,
        return state

    def _batch_state_random_initializer(self, idx_list=None):
        if idx_list is None:
            self.key, subkey = jax.random.split(self.key)
            batch_q = jax.random.uniform(
                subkey, (self.batch_size, 2), minval=-3.14, maxval=3.14)
            batch_qdot = jnp.zeros((self.batch_size, 2))
            self.key, subkey = jax.random.split(self.key)
            batch_target = jax.random.uniform(
                subkey,  (self.batch_size, 2), minval=-0.27, maxval=0.27)
            batch_to_target_vec = jax.vmap(jax.jit(partial(self.calc_pos_target)))(batch_target) - \
                jax.vmap(jax.jit(partial(self.calc_pos_fingertip)))(batch_q)
            state = jnp.concatenate(
                [batch_q, batch_qdot, batch_target, batch_to_target_vec[:, 0:2]], axis=1)
            return state
        else:
            idx_num = len(idx_list)
            self.key, subkey = jax.random.split(self.key)
            batch_q = jax.random.uniform(
                subkey, (idx_num, 2), minval=-3.14, maxval=3.14)
            batch_qdot = jnp.zeros((idx_num, 2))
            self.key, subkey = jax.random.split(self.key)
            batch_target = jax.random.uniform(
                subkey,  (idx_num, 2), minval=-0.27, maxval=0.27)
            batch_to_target_vec = jax.vmap(jax.jit(partial(self.calc_pos_target)))(batch_target) - \
                jax.vmap(jax.jit(partial(self.calc_pos_fingertip)))(batch_q)
            update_state = jnp.concatenate(
                [batch_q, batch_qdot, batch_target, batch_to_target_vec[:, 0:2]], axis=1)

            state = index_update(
                self.state,
                index[idx_list, :],
                update_state
            )
            return state

    def _step_fun(self, action):
        u = jnp.array(action)
        dynamics_params = (self.x_tree, self.inertia, u, self.a_grav)
        next_state = self.dynamics_step(
            self.dynamics_fun, self.state, *dynamics_params)
        done = self.done_fun(next_state)
        reward = self.reward_fun(self.state, action, next_state)
        self.state = next_state
        next_entry = (next_state, reward, done, {})
        return next_entry

    def _batch_step_fun(self, action):
        u = jnp.reshape(jnp.array(action), newshape=(self.batch_size, -1))
        dynamics_params = (self.x_tree, self.inertia, u, self.a_grav)
        next_state = jax.vmap(self.dynamics_step, (None, 0, None, None, 0, None), 0)(
            self.dynamics_fun, self.state, *dynamics_params)
        done = jax.vmap(self.done_fun)(next_state)
        reward = jax.vmap(self.reward_fun)(self.state, action, next_state)
        self.state = next_state
        next_entry = (next_state, reward, done, {})
        return next_entry

    def draw_line(self, from_pos, to_pos, line_color_rgb, line_width):
        if self.render:
            self.viewer_client.addUserDebugLine(
                from_pos, to_pos, line_color_rgb, line_width)


if __name__ == "__main__":

    env = Reacher(render=True)

    for i in range(1000):
        env.state = jnp.array([jnp.sin(i/100.0), jnp.cos(i/100.0), 0., 0.])
        env.reset_render_state()
