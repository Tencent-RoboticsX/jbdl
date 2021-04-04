import numpy as np
from jaxRBDL.Kinematics import calc_body_to_base_coordinates
from jaxRBDL.Kinematics.CalcPointVelocity import CalcPointVelocity
from typing import Tuple

def CalcPosVelPointToBase(model: dict, q: np.ndarray, qdot: np.ndarray, idbody: int, idbase: int, tarpoint: np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    
    pos_body = calc_body_to_base_coordinates(model, q, idbody, tarpoint)
    vel_body = CalcPointVelocity(model, q, qdot, idbody, tarpoint)

    pos_base = calc_body_to_base_coordinates(model, q, idbase, np.zeros((3, 1)))
    vel_base = CalcPointVelocity(model, q, qdot, idbase, np.zeros((3, 1)))

    pos = pos_body - pos_base
    vel = vel_body - vel_base

    return pos, vel