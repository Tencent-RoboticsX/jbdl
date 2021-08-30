from functools import partial
import math
import jax
from jbdl.envs.base_env import BaseEnv
import jax.numpy as jnp
import pybullet
from jbdl.envs.utils.parser import URDFBasedRobot
from jax.ops import index_update, index
from jbdl.rbdl.dynamics.forward_dynamics import forward_dynamics_core
from jbdl.experimental.ode.runge_kutta import odeint
from jbdl.rbdl.utils import xyz2int
from jbdl.rbdl.model.rigid_body_inertia import rigid_body_inertia
# from simulator.UrdfWrapper import UrdfWrapper
from jbdl.envs.utils.urdf_wrapper import UrdfWrapper
from jbdl.envs.utils.urdf_reader import URDF
from jbdl.envs.utils.urdf_utils import transform_origin
import os
from jbdl.envs import get_urdf_path

M_CART = 1.0
M_POLE = 0.1
HALF_POLE_LENGTH = 0.5
POLE_IC_PARAMS = jnp.zeros((6,))

LINK_MASS = []
LINK_INERTIA = []
LINK_COM = []
file_name = "arm.urdf"
file_path = os.path.join(get_urdf_path(), file_name)

robot = URDF.load(file_path)
for i in range(len(robot.links)):
    link_mass = robot.links[i].inertial.mass
    link_intertia = robot.links[i].inertial.inertia
    link_com = transform_origin(robot.links[i].inertial.origin)[0]
    LINK_MASS.append(link_mass)
    LINK_INERTIA.append(link_intertia)
    LINK_COM.append(link_com)

# DEFAULT_PURE_CART_POLE_PARAMS = (
#     M_CART, M_POLE, HALF_POLE_LENGTH, POLE_IC_PARAMS)

DEFAULT_PURE_ARM_PARAMS = (
    LINK_MASS, LINK_INERTIA, LINK_COM
)


class Arm(BaseEnv):
    def __init__(self, pure_arm_params=DEFAULT_PURE_ARM_PARAMS,
                 reward_fun=None,
                 seed=123, sim_dt=0.1, rtol=1.4e-8, atol=1.4e-8,
                 mxstep=jnp.inf, batch_size=0,
                 render=False, render_engine_name="pybullet", render_idx=None):

        self.model = UrdfWrapper(file_path).model
        # print(self.model)
        self.nb = self.model["nb"]
        self.nf = 3
        self.a_grav = self.model["a_grav"]
        self.jtype = tuple(self.model["jtype"])
        self.jaxis = xyz2int(self.model["jaxis"])
        self.parent = tuple(self.model["parent"])
        self.x_tree = self.model["x_tree"]
        self.sim_dt = sim_dt
        self.batch_size = batch_size

        # Angle at which to fail the episode
        # Angle at which to fail the episode
        self.theta_threshold = 5.0 / 360.0 * math.pi
        self.theta_done = 30.0 / 360.0 * math.pi
        self.x_threshold = 2.5
        self.qdot_threshold = 100.0
        self.target = jnp.array([0, 0, 0, 0, 1.57, 0, 0])
        self.key = jax.random.PRNGKey(seed)

        self.render = render
        self.render_engine_name = render_engine_name

        if render_engine_name == "pybullet":
            render_engine = pybullet
        elif render_engine_name == "xmirror":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        super().__init__(pure_arm_params, seed=seed,
                         sim_dt=sim_dt, rtol=rtol, atol=atol, mxstep=mxstep,
                         batch_size=batch_size, render=render,
                         render_engine=render_engine, render_idx=render_idx)

        def _dynamics_fun_core(y, t, x_tree, inertia, u, a_grav, parent, jtype, jaxis, nb):
            q = y[0:nb]
            qdot = y[nb:2*nb]
            input = (x_tree, inertia, parent, jtype, jaxis, nb, q, qdot, u, a_grav)
            qddot = forward_dynamics_core(*input)
            ydot = jnp.hstack([qdot, qddot])
            return ydot

        self._dynamics_fun = partial(
            _dynamics_fun_core, parent=self.parent, jtype=self.jtype, jaxis=self.jaxis, nb=self.nb)

        def _dynamics_step_core(dynamics_fun, y0, *args, nb, sim_dt, rtol, atol, mxstep):
            y_init = y0[0:2*nb]
            t_eval = jnp.linspace(0, sim_dt, 2)
            y_all = odeint(dynamics_fun, y_init, t_eval, *args,
                           rtol=rtol, atol=atol, mxstep=mxstep)
            y_final = y_all[-1, :]
            return y_final

        self._dynamics_step = partial(
            _dynamics_step_core, nb=self.nb, sim_dt=self.sim_dt, rtol=self.rtol, atol=self.atol, mxstep=self.mxstep)

        def _dynamics_step_with_params_core(
            dynamics_fun, state, action, *pure_arm_params, x_tree, a_grav):

            inertia = []
            link_mass_list, link_inertia_list, link_com_list = pure_arm_params
            for i in range(len(LINK_MASS)):
                link_mass = link_mass_list[i]
                link_com = link_com_list[i]
                link_intertia = link_inertia_list[i]
                inertia_element = rigid_body_inertia(link_mass, jnp.asarray(link_com), jnp.asarray(link_intertia))
                inertia.append(inertia_element)
            u = jnp.array([action[0], 0.0])
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

        def _done_fun(state, x_threshold=self.x_threshold, theta_threshold=self.theta_done):
            # x = state[0]
            # theta = state[1]
            qdot = state[self.nb:2*self.nb]
            done = jax.lax.cond(
                (len(qdot[qdot > self.qdot_threshold]) > 0),
                lambda done: True,
                lambda done: False,
                None)
            return done

        self.done_fun = jax.jit(_done_fun)

        def _default_reward_fun(state, action, next_state):
            q = state[0:self.nb]
            reward = - jnp.log(jnp.sum(jnp.square(q - self.target))) 
            return reward

        if reward_fun is None:
            self._reward_fun = _default_reward_fun
            self.reward_fun = jax.jit(_default_reward_fun)
        else:
            self._reward_fun = reward_fun
            self.reward_fun = jax.jit(reward_fun)

    def _init_pure_params(self, *pure_arm_params):
        # self.m_cart, self.m_pole, self.half_pole_length, self.pole_ic_params = pure_arm_pole_params
        # self.inertia_cart = self.init_inertia(
        #     self.m_cart, jnp.zeros((3,)), jnp.zeros((6,)))
        # self.inertia_pole = self.init_inertia(self.m_pole, jnp.array(
        #     [0.0, 0.0, self.half_pole_length]), self.pole_ic_params)
        # self.inertia = [self.inertia_cart, self.inertia_pole]
        inertia = []
        link_mass_list, link_inertia_list, link_com_list = pure_arm_params
        for i in range(len(LINK_MASS)):
            link_mass = link_mass_list[i]
            link_com = link_com_list[i]
            link_intertia = link_inertia_list[i]
            inertia_element = rigid_body_inertia(link_mass, jnp.asarray(link_com), jnp.asarray(link_intertia))
            inertia.append(inertia_element)
        self.inertia = inertia

    def _load_render_robot(self, viewer_client):
        render_robot = None
        if self.render_engine_name == "pybullet":
            viewer_client.connect(viewer_client.GUI)
            viewer_client.resetDebugVisualizerCamera(
                cameraDistance=6.18, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 1.0])
            render_robot = URDFBasedRobot(
                "cartpole.urdf", "physics", action_dim=1, obs_dim=4)
            render_robot.load(viewer_client)

        elif self.render_engine_name == "xmirror":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return render_robot

    def _get_render_state(self):
        if self.batch_size == 0:
            return (self.state[0], self.state[1])
        else:
            return (self.state[self.render_idx, 0], self.state[self.render_idx, 1])

    def _reset_render_state(self, *render_robot_state):
        x, theta = render_robot_state
        if self.render_engine_name == "pybullet":
            self.render_robot.jdict["slider_to_cart"].reset_current_position(
                x, 0)
            self.render_robot.jdict["cart_to_pole"].reset_current_position(
                theta, 0)
        elif self.render_engine_name == "xmirror":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

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


if __name__ == "__main__":
    env = Arm(render=True)
    q = jnp.zeros(7)
    qdot = jnp.zeros(7)
    state = jnp.array([q, qdot]).flatten()
    env.state = state
    env.reset_render_state()
