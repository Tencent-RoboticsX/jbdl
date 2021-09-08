
from os import stat
from jax.core import Value

from jbdl.experimental import contact
from jbdl.envs.base_env import BaseEnv
# from jbdlenvs.envs.reacher_env import IC_PARAMS_BODY0
from jbdl.envs.utils.parser import MJCFBasedRobot, URDFBasedRobot
import jax.numpy as jnp
from jbdl.rbdl.utils import xyz2int
from jbdl.rbdl.math import x_trans
import numpy as np
from jbdl.rbdl.dynamics.forward_dynamics import forward_dynamics_core
from jbdl.rbdl.dynamics.state_fun_ode import dynamics_fun_extend_core, events_fun_extend_core
from functools import partial
from jbdl.experimental.ode.runge_kutta import odeint
from jbdl.experimental.contact import detect_contact_core
import jax
from jbdl.experimental.contact.impulsive_dynamics import impulsive_dynamics_extend_core
from jbdl.rbdl.dynamics import composite_rigid_body_algorithm_core
from jbdl.experimental.ode.solve_ivp import solve_ivp
from jax.ops import index_update, index

M_TORSO = 1.0
M_THIGH = 1.0
M_LEG = 1.0
M_FOOT = 1.0

IC_PARAMS_TORSO = jnp.zeros((6,))
IC_PARAMS_THIGH = jnp.zeros((6,))
IC_PARAMS_LEG = jnp.zeros((6,))
IC_PARAMS_FOOT = jnp.zeros((6,))

DEFAULT_PURE_HOPPER_PARAMS = (
    M_TORSO,
    M_THIGH,
    M_LEG,
    M_FOOT,
    IC_PARAMS_TORSO,
    IC_PARAMS_THIGH,
    IC_PARAMS_LEG,
    IC_PARAMS_FOOT)


class Hopper(BaseEnv):
    def __init__(
            self,
            pure_hopper_params=DEFAULT_PURE_HOPPER_PARAMS,
            reward_fun=None,
            seed=123,
            sim_dt=2e-3,
            rtol=1.4e-8,
            atol=1.4e-8,
            mxstep=jnp.inf,
            batch_size=0,
            render=False,
            render_idx=None):

        self.nb = 6
        self.nc = 1
        self.nf = 2

        self.a_grav = jnp.array([[0.], [0.], [0.], [0.], [0.], [-9.81]])
        self.jtype = (1, 1, 0, 0, 0, 0)
        self.jaxis = xyz2int('xzyyyy')
        self.parent = tuple([0, 1, 2, 3, 4, 5])
        self.x_tree = list([jnp.eye(6) for i in range(self.nb)])
        self.id_contact = (6,)
        # self.contact_point = (jnp.array([[0.], [0.], [0]]))
        self.contact_point = (jnp.array([0., 0., 0.]))
        self.contact_force_lb = jnp.array([-1000.0, -1000.0, 0.0])
        self.contact_force_ub = jnp.array([1000.0, 1000.0, 3000.0])
        self.contact_pos_lb = jnp.array([0.0001, 0.0001, 0.0001])
        self.contact_pos_ub = jnp.array([0.0001, 0.0001, 0.0001])
        self.contact_vel_lb = jnp.array([-0.05, -0.05, -0.05])
        self.contact_vel_ub = jnp.array([0.01, 0.01, 0.01])
        self.mu = 0.9
        self.st = jnp.vstack([jnp.zeros((3, 3)), jnp.identity(3)])

        self.sim_dt = 0.1
        self.batch_size = batch_size

        self.render_engine_name = "pybullet"

        super().__init__(
            pure_hopper_params,
            seed=seed,
            sim_dt=sim_dt,
            rtol=rtol,
            atol=atol,
            mxstep=mxstep,
            batch_size=batch_size,
            render=render,
            render_idx=render_idx)

        def _dynamics_fun_core(
            y,
            t,
            x_tree,
            inertia,
            u,
            a_grav,
            contact_point,
            contact_force_lb,
            contact_force_ub,
            contact_pos_lb,
            contact_vel_lb,
            contact_vel_ub,
            mu,
            st,
            id_contact,
            parent,
            jtype,
            jaxis,
            nb,
            nc,
            nf,
                ncp):
            q = y[0:nb]
            qdot = y[nb:2 * nb]
            tau = jnp.matmul(st, u)
            # tau = u
            flag_contact = detect_contact_core(
                x_tree,
                q,
                qdot,
                contact_point,
                contact_pos_lb,
                contact_vel_lb,
                contact_vel_ub,
                id_contact,
                parent,
                jtype,
                jaxis,
                nc)
            ydot, fqp, h = dynamics_fun_extend_core(
                x_tree, inertia, q, qdot, contact_point, tau, a_grav,
                contact_force_lb, contact_force_ub, id_contact, flag_contact,
                parent, jtype, jaxis, nb, nc, nf, ncp, mu)
            return ydot

        def _events_fun_core(
            y,
            t,
            x_tree,
            inertia,
            u,
            a_grav,
            contact_point,
            contact_force_lb,
            contact_force_ub,
            contact_pos_lb,
            contact_vel_lb,
            contact_vel_ub,
            mu,
            st,
            id_contact,
            parent,
            jtype,
            jaxis,
            nb,
            nc,
            nf,
                ncp):
            q = y[0:nb]
            qdot = y[nb:2 * nb]
            flag_contact = detect_contact_core(
                x_tree,
                q,
                qdot,
                contact_point,
                contact_pos_lb,
                contact_vel_lb,
                contact_vel_ub,
                id_contact,
                parent,
                jtype,
                jaxis,
                nc)
            value = events_fun_extend_core(
                x_tree,
                q,
                contact_point,
                id_contact,
                flag_contact,
                parent,
                jtype,
                jaxis,
                nc)
            return value

        def _impulsive_dynamics_fun_core(
            y,
            t,
            x_tree,
            inertia,
            u,
            a_grav,
            contact_point,
            contact_force_lb,
            contact_force_ub,
            contact_pos_lb,
            contact_vel_lb,
            contact_vel_ub,
            mu,
            st,
            id_contact,
            parent,
            jtype,
            jaxis,
            nb,
            nc,
            nf,
                ncp):
            q = y[0:nb]
            qdot = y[nb:2 * nb]
            h = composite_rigid_body_algorithm_core(
                x_tree, inertia, parent, jtype, jaxis, nb, q)
            flag_contact = detect_contact_core(
                x_tree,
                q,
                qdot,
                contact_point,
                contact_pos_lb,
                contact_vel_lb,
                contact_vel_ub,
                id_contact,
                parent,
                jtype,
                jaxis,
                nc)
            qdot_impulse = impulsive_dynamics_extend_core(
                x_tree,
                q,
                qdot,
                contact_point,
                h,
                id_contact,
                flag_contact,
                parent,
                jtype,
                jaxis,
                nb,
                nc,
                nf)
            qdot_impulse = qdot_impulse.flatten()
            y_new = jnp.hstack([q, qdot_impulse])
            return y_new

        def _fqp_fun_core(
            y,
            t,
            x_tree,
            inertia,
            u,
            a_grav,
            contact_point,
            contact_force_lb,
            contact_force_ub,
            contact_pos_lb,
            contact_vel_lb,
            contact_vel_ub,
            mu,
            st,
            id_contact,
            parent,
            jtype,
            jaxis,
            nb,
            nc,
            nf,
                ncp):
            q = y[0:nb]
            qdot = y[nb:2 * nb]
            tau = jnp.matmul(st, u)
            flag_contact = detect_contact_core(
                x_tree,
                q,
                qdot,
                contact_point,
                contact_pos_lb,
                contact_vel_lb,
                contact_vel_ub,
                id_contact,
                parent,
                jtype,
                jaxis,
                nc)
            xdot, fqp, h = dynamics_fun_extend_core(x_tree, inertia, q, qdot, contact_point, tau, a_grav, 
                                                    contact_force_lb, contact_force_ub, id_contact, 
                                                    flag_contact, parent, jtype, jaxis, 
                                                    nb, nc, nf, ncp, mu)
            return fqp, flag_contact

        self._dynamics_fun = partial(_dynamics_fun_core,
                                     st=self.st,
                                     id_contact=self.id_contact,
                                     parent=self.parent,
                                     jtype=self.jtype,
                                     jaxis=self.jaxis,
                                     nb=self.nb,
                                     nc=self.nc,
                                     nf=self.nf,
                                     ncp=0.0)

        self._events_fun = partial(_events_fun_core,
                                   st=self.st,
                                   id_contact=self.id_contact,
                                   parent=self.parent,
                                   jtype=self.jtype,
                                   jaxis=self.jaxis,
                                   nb=self.nb,
                                   nc=self.nc,
                                   nf=self.nf,
                                   ncp=0.0)

        self._impulsive_dynamics_fun = partial(_impulsive_dynamics_fun_core,
                                               st=self.st,
                                               id_contact=self.id_contact,
                                               parent=self.parent,
                                               jtype=self.jtype,
                                               jaxis=self.jaxis,
                                               nb=self.nb,
                                               nc=self.nc,
                                               nf=self.nf,
                                               ncp=0.0)

        self._fqp_fun = partial(_fqp_fun_core,
                                st=self.st,
                                id_contact=self.id_contact,
                                parent=self.parent,
                                jtype=self.jtype,
                                jaxis=self.jaxis,
                                nb=self.nb,
                                nc=self.nc,
                                nf=self.nf,
                                ncp=0.0)

        def _dynamics_step_core(
                dynamics_fun,
                events_fun,
                impulsive_dynamics_fun,
                y0,
                *args,
                nb,
                sim_dt,
                rtol,
                atol,
                mxstep):
            y_init = y0[0:2 * nb]
            t_eval = jnp.linspace(0, sim_dt, 2)
            y_all = solve_ivp(
                dynamics_fun,
                y_init,
                t_eval,
                events_fun,
                impulsive_dynamics_fun,
                *args,
                rtol=rtol,
                atol=atol,
                mxstep=mxstep)
            y_final = y_all[-1, :]

            yT = y_final
            return yT

        self._dynamics_step = partial(
            _dynamics_step_core,
            nb=self.nb,
            sim_dt=self.sim_dt,
            rtol=self.rtol,
            atol=self.atol,
            mxstep=self.mxstep)

        self.dynamics_fun = jax.jit(self._dynamics_fun)
        self.events_fun = jax.jit(self._events_fun)
        self.impulsive_dynamics_fun = jax.jit(self._impulsive_dynamics_fun)
        self.fqp_fun = jax.jit(self._fqp_fun)
        self.dynamics_step = self._dynamics_step

        def _done_fun(next_state):
            return False
        self.done_fun = jax.jit(_done_fun)

        def _default_reward_fun(state, action, next_state):
            return 1.0

        if reward_fun is None:
            self._reward_fun = _default_reward_fun
            self.reward_fun = jax.jit(_default_reward_fun)
        else:
            self._reward_fun = reward_fun
            self.reward_fun = jax.jit(reward_fun)

    def _init_pure_params(self, *pure_env_params):
        self.m_torso, self.m_thigh, self.m_leg, self.m_foot, \
            self.IC_params_torso, self.IC_params_thigh, self.IC_params_leg, self.IC_params_foot = pure_env_params

        self.I_rootx = jnp.zeros((6, 6))
        self.I_rootz = jnp.zeros((6, 6))
        self.I_rooty = self.init_inertia(self.m_torso, jnp.array(
            [0.00000, 0.00000, 0.2000]), self.IC_params_torso)
        self.I_m_thigh = self.init_inertia(self.m_thigh, jnp.array(
            [0.00000, 0.00000, 0.22500]), self.IC_params_thigh)
        self.I_m_leg = self.init_inertia(self.m_leg, jnp.array(
            [0.00000, 0.00000, 0.25000]), self.IC_params_leg)
        self.I_m_foot = self.init_inertia(self.m_foot, jnp.array(
            [0.06500, 0.00000, 0.19500]), self.IC_params_foot)

        self.inertia = [
            self.I_rootx,
            self.I_rootz,  
            self.I_rooty,          
            self.I_m_thigh,
            self.I_m_leg,
            self.I_m_foot]

    def _load_render_robot(self, pybullet_client):
        pybullet_client.connect(pybullet_client.GUI)
        robot = MJCFBasedRobot(
            "hopper.xml",
            "torso",
            action_dim=6,
            obs_dim=12)

        robot.load(pybullet_client)
        pybullet_client.resetDebugVisualizerCamera(
            cameraDistance=4.0,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[
                0,
                0,
                0])
        print(robot.jdict)
        return robot

    def _reset_render_state(self, *render_robot_state):
        rootx, rootz, rooty, thigh_joint, leg_joint, foot_joint = render_robot_state
        
        self.render_robot.jdict["rootx"].reset_current_position(rootx, 0)
        self.render_robot.jdict["rootz"].reset_current_position(rootz, 0)
        self.render_robot.jdict["rooty"].reset_current_position(rooty, 0)
        self.render_robot.jdict["thigh_joint"].reset_current_position(
            thigh_joint, 0)
        self.render_robot.jdict["leg_joint"].reset_current_position(
            leg_joint, 0)
        self.render_robot.jdict["foot_joint"].reset_current_position(
            foot_joint, 0)

    def _get_render_state(self):
        if self.batch_size == 0:
            return (
                self.state[0],
                self.state[1],
                self.state[2],
                self.state[3],
                self.state[4],
                self.state[5],
                )
        else:
            return (
                self.state[self.render_idx, 0],
                self.state[self.render_idx, 1],
                self.state[self.render_idx, 2],
                self.state[self.render_idx, 3],
                self.state[self.render_idx, 4]
            )

    def _state_random_initializer(self):
        self.key, subkey = jax.random.split(self.key)
        q = jax.random.uniform(subkey, shape=(6,), minval=-0.1, maxval=0.1)
        qdot = jnp.zeros((6,))
        state = jnp.concatenate([q, qdot])
        return state

    def _batch_state_random_initializer(self, idx_list):
        if idx_list is None:
            self.key, subkey = jax.random.split(self.key)
            batch_q = jax.random.uniform(
                subkey, (self.batch_size, 9), minval=-0.1, maxval=0.1)
            batch_qdot = jnp.zeros((self.batch_size, 9))
            state = jnp.concatenate([batch_q, batch_qdot], axis=1)
            return state
        else:
            idx_num = len(idx_list)
            self.key, subkey = jax.random.split(self.key)
            batch_q = jax.random.uniform(
                subkey, (idx_num, 9), minval=-0.1, maxval=0.1)
            batch_qdot = jnp.zeros((idx_num, 2))
            update_state = jnp.concatenate([batch_q, batch_qdot], axis=1)

            state = index_update(
                self.state,
                index[idx_list, :],
                update_state
            )
            return state

    def _step_fun(self, action):
        u = jnp.array(action)
        dynamics_params = (
            self.x_tree,
            self.inertia, u,
            self.a_grav,
            self.contact_point,
            self.contact_force_lb,
            self.contact_force_ub,
            self.contact_pos_lb,
            self.contact_vel_lb,
            self.contact_vel_ub,
            self.mu)
        next_state = self.dynamics_step(
            self.dynamics_fun,
            self.events_fun,
            self.impulsive_dynamics_fun,
            self.state,
            *dynamics_params)
        done = self.done_fun(next_state)
        reward = self.reward_fun(self.state, action, next_state)
        self.state = next_state
        next_entry = (next_state, reward, done, {})
        return next_entry

    def _batch_step_fun(self, action):
        u = jnp.reshape(jnp.array(action), newshape=(self.batch_size, -1))
        dynamics_params = (
            self.x_tree,
            self.inertia, u,
            self.a_grav,
            self.contact_point,
            self.contact_force_lb,
            self.contact_force_ub,
            self.contact_pos_lb,
            self.contact_vel_lb,
            self.contact_vel_ub,
            self.mu)
        next_state = jax.vmap(self.dynamics_step,
                              (None,
                               None,
                               None,
                               0,
                               None,
                               None,
                               0,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               None),
                              0)(self.dynamics_fun,
                                 self.events_fun,
                                 self.impulsive_dynamics_fun,
                                 self.state,
                                 *dynamics_params)
        done = jax.vmap(self.done_fun)(next_state)
        reward = jax.vmap(self.reward_fun)(self.state, action, next_state)
        self.state = next_state
        next_entry = (next_state, reward, done, {})
        return next_entry


if __name__ == "__main__":
    import time
    env = Hopper(render=True)
    state = env.state
    env.reset_render_state()
    time.sleep(10)
