import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from jbdl.rbdl.kinematics import calc_body_to_base_coordinates
from jbdl.rbdl.tools import clac_inertia_cuboid
from jbdl.rbdl.tools.plot_inertia_cuboid import plot_inertia_cuboid


def plot_com_inertia(model: dict, q: np.ndarray, ax: Axes3D):

    idcomplot = model["idcomplot"]
    com = model["CoM"]
    inertia = model["Inertia"]
    mass = model["Mass"]

    num = len(idcomplot)
    pos_com = []
    for i in range(num):
        pos_com.append(calc_body_to_base_coordinates(model, q, idcomplot[i], com[i]))
        
    pos_com = np.asfarray(np.concatenate(pos_com, axis=1))
    ax.scatter(pos_com[0, :], pos_com[1, :], pos_com[2, :], marker="*")

    for i in range(num):
        lxyz  = clac_inertia_cuboid(np.diag(inertia[i]), mass[i])
        plot_inertia_cuboid(pos_com[:, i:i+1], lxyz, ax)

    return ax
