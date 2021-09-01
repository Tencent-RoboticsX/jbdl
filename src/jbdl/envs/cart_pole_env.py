from functools import partial
#import os
import math
import jax
from jbdl.envs.base_env import BaseEnv
import jax.numpy as jnp
import pybullet
import meshcat
from jbdl.envs.utils.parser import URDFBasedRobot
from jax.ops import index_update, index
from jbdl.rbdl.dynamics.forward_dynamics import forward_dynamics_core
from jbdl.experimental.ode.runge_kutta import odeint
from jbdl.rbdl.utils import xyz2int
#from jbdl.envs import get_urdf_path
from jbdl.experimental.render.xmirror import robot
from jbdl.experimental.render.xmirror.visual import Pos

M_CART = 1.0
M_POLE = 0.1
HALF_POLE_LENGTH = 0.5
POLE_IC_PARAMS = jnp.zeros((6,))
DEFAULT_PURE_CART_POLE_PARAMS = (
    M_CART, M_POLE, HALF_POLE_LENGTH, POLE_IC_PARAMS)


class CartPole(BaseEnv):
    def __init__(self, pure_cart_pole_params=DEFAULT_PURE_CART_POLE_PARAMS, reward_fun=None,
                 seed=123, sim_dt=0.1, rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf, batch_size=0,
                 render=False, render_engine_name="pybullet", render_idx=None):

        self.nb = 2
        self.nf = 3
        self.a_grav = jnp.array([[0.], [0.], [0.], [0.], [0.], [-9.81]])
        self.jtype = (1, 0)
        self.jaxis = xyz2int('xy')
        self.parent = (0, 1)
        self.x_tree = list([jnp.eye(6) for i in range(self.nb)])
        self.sim_dt = sim_dt
        self.batch_size = batch_size

        # Angle at which to fail the episode
        # Angle at which to fail the episode
        self.theta_threshold = 5.0 / 360.0 * math.pi
        self.theta_done = 30.0 / 360.0 * math.pi
        self.x_threshold = 2.5
        self.key = jax.random.PRNGKey(seed)

        self.render = render
        self.render_engine_name = render_engine_name

        if render_engine_name == "pybullet":
            render_engine = pybullet
        elif render_engine_name == "xmirror":
            render_engine = meshcat.Visualizer()  # TODO wrap it to xmirror
        else:
            raise NotImplementedError()

        super().__init__(pure_cart_pole_params, seed=seed,
                         sim_dt=sim_dt, rtol=rtol, atol=atol, mxstep=mxstep,
                         batch_size=batch_size, render=render,
                         render_engine=render_engine, render_idx=render_idx)

        def _dynamics_fun_core(y, t, x_tree, inertia, u, a_grav, parent, jtype, jaxis, nb):
            q = y[0:nb]
            qdot = y[nb:2 * nb]
            input = (x_tree, inertia, parent, jtype, jaxis, nb, q, qdot, u, a_grav)
            qddot = forward_dynamics_core(*input)
            ydot = jnp.hstack([qdot, qddot])
            return ydot

        self._dynamics_fun = partial(
            _dynamics_fun_core, parent=self.parent, jtype=self.jtype, jaxis=self.jaxis, nb=self.nb)

        def _dynamics_step_core(dynamics_fun, y0, *args, nb, sim_dt, rtol, atol, mxstep):
            y_init = y0[0:2 * nb]
            t_eval = jnp.linspace(0, sim_dt, 2)
            y_all = odeint(dynamics_fun, y_init, t_eval, *args,
                           rtol=rtol, atol=atol, mxstep=mxstep)
            y_final = y_all[-1, :]
            return y_final

        self._dynamics_step = partial(
            _dynamics_step_core, nb=self.nb, sim_dt=self.sim_dt, rtol=self.rtol, atol=self.atol, mxstep=self.mxstep)

        def _dynamics_step_with_params_core(
                dynamics_fun, state, action, *pure_cart_pole_params, x_tree, a_grav):

            m_cart, m_pole, half_pole_length, pole_ic_params = pure_cart_pole_params
            inertia_cart = self.init_inertia(
                m_cart, jnp.zeros((3,)), jnp.zeros((6,)))
            inertia_pole = self.init_inertia(m_pole, jnp.array(
                [0.0, 0.0, half_pole_length]), pole_ic_params)
            inertia = [inertia_cart, inertia_pole]
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
            x = state[0]
            theta = state[1]
            done = jax.lax.cond(
                (jnp.abs(x) > jnp.abs(x_threshold)) +
                (jnp.abs(theta) > jnp.abs(theta_threshold)),
                lambda done: True,
                lambda done: False,
                None)
            return done

        self.done_fun = jax.jit(_done_fun)

        def _default_reward_fun(state, action, next_state):
            reward = -(next_state[0] ** 2 + 10 * next_state[1]
                       ** 2 + next_state[2] ** 2 + next_state[3] ** 2)
            return reward

        if reward_fun is None:
            self._reward_fun = _default_reward_fun
            self.reward_fun = jax.jit(_default_reward_fun)
        else:
            self._reward_fun = reward_fun
            self.reward_fun = jax.jit(reward_fun)

    def _init_pure_params(self, *pure_cart_pole_params):
        self.m_cart, self.m_pole, self.half_pole_length, self.pole_ic_params = pure_cart_pole_params
        self.inertia_cart = self.init_inertia(
            self.m_cart, jnp.zeros((3,)), jnp.zeros((6,)))
        self.inertia_pole = self.init_inertia(self.m_pole, jnp.array(
            [0.0, 0.0, self.half_pole_length]), self.pole_ic_params)
        self.inertia = [self.inertia_cart, self.inertia_pole]

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
            viewer_client.open()
            # no camera set yet
            # urdf_path = os.path.join(get_urdf_path(), "cartpole.urdf")
            # render_robot = robot.RobotModel(vis=viewer_client, name="cart_pole", id=1, xml_path=urdf_path)
            render_robot = robot.RobotModel(vis=viewer_client, name="cart_pole", id=1)
            render_robot.link_name_tree.name_tree = {"cart_pole": {"slideBar": {"cart": {"pole": {}}}}}
            link1 = robot.Link(viewer_client,
                               "cart_pole/slideBar",
                               id=0, pos=Pos(),
                               geom=[{"box": [30, 0.05, 0.05]}],
                               visual_pos={"sliderBar1": Pos()})
            render_robot.links.append(link1)
            link2 = robot.Link(viewer_client,
                               "cart_pole/slideBar/cart",
                               id=1, pos=Pos(),
                               geom=[{"box": [0.5, 0.5, 0.2]}],
                               visual_pos={"cart1": Pos()})
            render_robot.links.append(link2)
            link3 = robot.Link(viewer_client,
                               "cart_pole/slideBar/cart/pole",
                               id=2, pos=Pos(),
                               geom=[{"box": [0.05, 0.05, 1.0]}],
                               visual_pos={"pole1": Pos(xyz=[0, 0, 0.5])})
            render_robot.links.append(link3)
            joint1 = robot.Joint(viewer_client,
                                 name="slider_to_cart",
                                 id=0, pos=Pos(),
                                 type="prismatic",
                                 parent_link="sildeBar",
                                 child_link="cart",
                                 axis=[1, 0, 0])
            render_robot.joints.append(joint1)
            joint2 = robot.Joint(viewer_client,
                                 name="cart_to_pole",
                                 id=0, pos=Pos(),
                                 type="continuous",
                                 parent_link="cart",
                                 child_link="pole",
                                 axis=[0, 1, 0])
            render_robot.joints.append(joint2)
            render_robot.render()
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
            self.render_robot.set_joint_state(joint_name="slider_to_cart", state=x)
            self.render_robot.set_joint_state(joint_name="cart_to_pole", state=theta)
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
        u = jnp.array([action[0], 0.0])
        dynamics_params = (self.x_tree, self.inertia, u, self.a_grav)
        next_state = self.dynamics_step(
            self.dynamics_fun, self.state, *dynamics_params)
        done = self.done_fun(next_state)
        reward = self.reward_fun(self.state, action, next_state)
        self.state = next_state
        next_entry = (next_state, reward, done, {})
        return next_entry

    def _batch_step_fun(self, action):
        action = jnp.reshape(jnp.array(action), newshape=(self.batch_size, -1))
        u = jnp.hstack([action, jnp.zeros((self.batch_size, 1))])
        dynamics_params = (self.x_tree, self.inertia, u, self.a_grav)
        next_state = jax.vmap(self.dynamics_step, (None, 0, None, None, 0, None), 0)(
            self.dynamics_fun, self.state, *dynamics_params)
        done = jax.vmap(self.done_fun)(next_state)
        reward = jax.vmap(self.reward_fun)(self.state, action, next_state)
        self.state = next_state
        next_entry = (next_state, reward, done, {})
        return next_entry


if __name__ == "__main__":
    env = CartPole(render=True, render_engine_name="xmirror")

    for i in range(1000):
        env.state = jnp.array([jnp.sin(i / 100.0), jnp.cos(i / 100.0), 0., 0.])
        env.reset_render_state()
