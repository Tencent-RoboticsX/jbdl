from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



def plot_link(pos_o, pos_e, pos_num, contact_point, ax: Axes3D):
    ax.scatter(pos_o[0,:], pos_o[1, :], pos_o[2, :], marker="o",color="r")
    for i in range(pos_num):
        ax.plot([pos_o[0, i], pos_e[0, i]],[pos_o[1, i], pos_e[1, i]],[pos_o[2, i],pos_e[2, i]])
    ax.scatter(contact_point[0, :], contact_point[1, :], contact_point[2, :], marker="o", color="c")
    return ax
    

if __name__ == "__main__":
    print("======")
    pos_o = np.array([[6.375e-27, 6.375e-27, 0.196, -0.196, 0.09025, -0.09025],
                      [0.0,	0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.44992, 0.44992, 0.44992, 0.44992, 0.26676,	0.26676]])
    pos_e = np.array([[0.196, -0.196, 0.09025, -0.09025, 0.22275, -0.22275],
                      [0.0,	0.0, 0.0, 0.0, 0.0,	0.0],
                      [0.44992, 0.44992, 0.26676, 0.26676, 0.03726, 0.03726]])

    pos_contact = np.array([[0.22275, -0.22275],
                             [0.0, 0.0],
                             [0.03726, 0.03726]])
    fig = plt.gcf()
    ax = Axes3D(fig)
    plot_link(pos_o, pos_e, 6, pos_contact, ax)
    plt.show()