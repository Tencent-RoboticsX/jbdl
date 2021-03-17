from matplotlib.pyplot import axis
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.testing._private.utils import import_nose
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from jaxRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates
from jaxRBDL.Tools.PlotLink import PlotLink

def PlotModel(model: dict, q: np.ndarray, ax: Axes3D):

    idlinkplot = model["idlinkplot"]
    linkplot = model["linkplot"]
    idcontact = model["idcontact"]
    contactpoint = model["contactpoint"]

    pos_o = []
    pos_e = []

    num = len(idlinkplot)

    for i in range(num):
        pos_o.append(CalcBodyToBaseCoordinates(model, q, idlinkplot[i], np.zeros((3,1))))
        pos_e.append(CalcBodyToBaseCoordinates(model, q, idlinkplot[i], linkplot[i]))

    pos_o = np.concatenate(pos_o, axis=1)
    pos_e = np.concatenate(pos_e, axis=1)

    nc = len(idcontact)

    pos_contact = []
    for i in range(nc):
        pos_contact.append(CalcBodyToBaseCoordinates(model, q, idcontact[i], contactpoint[i]))
    pos_contact = np.concatenate(pos_contact, axis=1)
    
    ax = PlotLink(pos_o, pos_e, num, pos_contact, ax)
    # ax = PlotLink(pos_o, pos_e, 16, pos_contact, ax)
    return ax

