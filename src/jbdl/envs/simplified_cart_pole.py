import math
from collections import namedtuple
from jbdl.envs.base_env import BaseEnv
from gym.envs.classic_control import rendering
import jax.numpy as jnp

M_CART = 1.0
M_POLE = 0.1
HALF_POLE_LENGTH = 0.5
DEFAULT_PURE_SIMPLIFIED_CART_POLE_PARAMS = (M_CART, M_POLE, HALF_POLE_LENGTH)

class SimplifiedCartPole(BaseEnv):
    def __init__(self, pure_simplifed_cart_pole_params=DEFAULT_PURE_SIMPLIFIED_CART_POLE_PARAMS,
                 seed=123, sim_dt=0.01, batch_size=0, render=False,
                 screen_width=600, screen_height=400, render_idx=None):

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
            render=render, render_engine=rendering.Viewer(screen_width, screen_height),
            render_idx=render_idx)


        def _dynamics_fun_core(pos_vel, action,  m_car, f_unit, a_gravity):
            x, theta, x_dot, theta_dot = pos_vel
            force = (2.0 * action - 1.0) * f_unit
            costheta = jnp.cos(theta)
            sintheta = jnp.sin(theta)

            # For the interested reader:
            # https://coneural.org/florian/papers/05_cart_pole.pdf
            temp = (
                force + self.polemass_length * theta_dot ** 2 * sintheta
            ) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
            )
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass


    def _init_pure_params(self, *pure_simplified_cart_pole_params):
        self.m_cart, self.m_pole, self.half_pole_length = pure_simplified_cart_pole_params
    
    def _load_render_robot(self, viewer_client):
        polewidth = 10.0
        polelen = self.scale * (2 * self.half_pole_length)
        cartwidth =  50.0
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
        track = rendering.Line((0, self.carty), (self.screen_width, self.carty))
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
        return super()._state_random_initializer()

    def _batch_state_random_initializer(self, idx_list):
        return super()._batch_state_random_initializer(idx_list)

    def _step_fun(self, action):
        return super()._step_fun(action)
    
    def _batch_step_fun(self, action):
        return super()._batch_step_fun(action)

    



if __name__ == "__main__":
    import numpy as np
    env = SimplifiedCartPole(render=True)
    env.state = jnp.zeros((4,))
    env.reset_render_state()
    for i in range(1000):
        env.state = jnp.array([0.0, np.sin(i/100), 0.0, 0.0])
        env.reset_render_state()

