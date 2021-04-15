from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from jaxBDL.rbdl.kinematics import calc_body_to_base_coordinates
from jaxBDL.rbdl.tools import clac_inertia_cuboid
from jaxBDL.rbdl.tools.plot_inertia_cuboid import plot_inertia_cuboid

def plot_com_inertia(model: dict, q: np.ndarray, ax: Axes3D):


    idcomplot = model["idcomplot"]
    CoM = model["CoM"]
    Inertia = model["Inertia"]
    Mass = model["Mass"]

    num = len(idcomplot)
    pos_com = []
    for i in range(num):
        pos_com.append(calc_body_to_base_coordinates(model, q, idcomplot[i], CoM[i]))
        
    pos_com = np.asfarray(np.concatenate(pos_com, axis=1))
    ax.scatter(pos_com[0,:], pos_com[1, :], pos_com[2, :], marker="*")

    for i in range(num):
        lxyz  = clac_inertia_cuboid(np.diag(Inertia[i]), Mass[i])
        plot_inertia_cuboid(pos_com[:, i:i+1], lxyz, ax)

    return ax