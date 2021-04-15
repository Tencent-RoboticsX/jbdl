from matplotlib.pyplot import axis
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.testing._private.utils import import_nose
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from jaxBDL.rbdl.kinematics import calc_body_to_base_coordinates
from jaxBDL.rbdl.tools import plot_link

def plot_model(model: dict, q: np.ndarray, ax: Axes3D):

    idlinkplot = model["idlinkplot"]
    linkplot = model["linkplot"]
    idcontact = model["idcontact"]
    contactpoint = model["contactpoint"]

    pos_o = []
    pos_e = []

    num = len(idlinkplot)

    for i in range(num):
        pos_o.append(calc_body_to_base_coordinates(model, q, idlinkplot[i], np.zeros((3,1))))
        pos_e.append(calc_body_to_base_coordinates(model, q, idlinkplot[i], linkplot[i]))

    pos_o = np.concatenate(pos_o, axis=1)
    pos_e = np.concatenate(pos_e, axis=1)

    nc = len(idcontact)

    pos_contact = []
    for i in range(nc):
        pos_contact.append(calc_body_to_base_coordinates(model, q, idcontact[i], contactpoint[i]))
    pos_contact = np.concatenate(pos_contact, axis=1)
    
    ax = plot_link(pos_o, pos_e, num, pos_contact, ax)
    # ax = PlotLink(pos_o, pos_e, 16, pos_contact, ax)
    return ax

