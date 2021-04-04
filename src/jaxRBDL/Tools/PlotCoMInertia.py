from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from jaxRBDL.Kinematics import calc_body_to_base_coordinates
from jaxRBDL.Tools.CalcInertiaCuboid import ClacInertiaCuboid
from jaxRBDL.Tools.PlotInertiaCuboid import PlotInertiaCuboid

def PlotCoMInertia(model: dict, q: np.ndarray, ax: Axes3D):


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

    # print(num)
    for i in range(num):
        lxyz  = ClacInertiaCuboid(np.diag(Inertia[i]), Mass[i])
        PlotInertiaCuboid(pos_com[:, i:i+1], lxyz, ax)

    return ax