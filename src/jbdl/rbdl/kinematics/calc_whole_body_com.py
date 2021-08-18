import numpy as np
from jbdl.rbdl.kinematics import calc_body_to_base_coordinates


def calc_whole_body_com(model: dict, q: np.ndarray) -> np.ndarray:
    """calc_whole_body_com - Calculate whole body's CoM position in world frame

    Args:
        model (dict): dictionary of model specification
        q (np.ndarray): an array of joint position

    Returns:
        np.ndarray: float (3, 3)
    """

    q = q.flatten()
    idcomplot = model["idcomplot"]
    com = model["com"]
    mass = model["mass"]

    num = len(idcomplot)
    com_list = []
    clink = np.zeros((3, 1))
    for i in range(num):
        clink = calc_body_to_base_coordinates(model, q, idcomplot[i], com[i])
        com_list.append(clink)

    c = np.zeros((3, 1))
    m = 0

    for i in range(num):
        c = c + np.multiply(com_list[i], mass[i])
        m = m + mass[i]

    pcom = np.asfarray(np.divide(c, m))

    return pcom
