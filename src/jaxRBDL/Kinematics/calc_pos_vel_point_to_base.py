import numpy as np
from jaxRBDL.Kinematics import calc_body_to_base_coordinates
from jaxRBDL.Kinematics import calc_point_velocity
from typing import Tuple

def calc_pos_vel_point_to_base(model: dict, q: np.ndarray, qdot: np.ndarray, idbody: int, idbase: int, tarpoint: np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    
    pos_body = calc_body_to_base_coordinates(model, q, idbody, tarpoint)
    vel_body = calc_point_velocity(model, q, qdot, idbody, tarpoint)

    pos_base = calc_body_to_base_coordinates(model, q, idbase, np.zeros((3, 1)))
    vel_base = calc_point_velocity(model, q, qdot, idbase, np.zeros((3, 1)))

    pos = pos_body - pos_base
    vel = vel_body - vel_base

    return pos, vel