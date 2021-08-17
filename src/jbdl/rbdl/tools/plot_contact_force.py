from typing import Optional
from jbdl.rbdl.kinematics import calc_body_to_base_coordinates
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np


def plot_contact_force(model: dict, q: np.ndarray, fc: Optional[np.ndarray] , fcqp: Optional[np.ndarray], fcpd: Optional[np.ndarray], fwho: str, ax: Axes3D):
    if fc is None and fcqp is None and fcpd is None: 
        print("No contact force to plot!")
        return
    
    fplot = None
    if fwho == "fc" and fc is not None:
        fplot = fc.flatten()
    if fwho == "fcqp" and fcqp is not None:
        fplot = fcqp.flatten()
    if fwho == "fcpd" and fcpd is not None:
        fplot = fcpd.flatten()
    if fplot is None:
        print("No contact force available!")
        return


    nc = model["nc"]
    idcontact = model["idcontact"]
    contactpoint = model["contactpoint"]

    fshrink = 100.0
    nc = len(idcontact)
    pos_contact_list = []
    fplot = fplot.reshape((3, nc), order="F")
    for i in range(nc):
        pos_contact_list.append(calc_body_to_base_coordinates(model, q, idcontact[i], contactpoint[i]))

    pos_contact = np.asfarray(np.concatenate(pos_contact_list, axis=1))
    # print(fplot)
    ax.quiver(pos_contact[0, :], pos_contact[1, :], pos_contact[2, :], fplot[0, :]/fshrink, fplot[1, :]/fshrink, fplot[2,:]/fshrink)



    
    


    