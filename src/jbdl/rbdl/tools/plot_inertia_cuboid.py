import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_cuboid(center, size, ax, color, alpha):
    """
       Create a data array for cuboid plotting.


       ============= ================================================
       Argument      Description
       ============= ================================================
       center        center of the cuboid, triple
       size          size of the cuboid, triple, (x_length,y_width,z_height)
       :type size: tuple, numpy.array, list
       :param size: size of the cuboid, triple, (x_length,y_width,z_height)
       :type center: tuple, numpy.array, list
       :param center: center of the cuboid, triple, (x,y,z)
   """
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    ox, oy, oz = center
    l, w, h = size

    x = np.linspace(ox-l/2, ox+l/2, num=10)
    y = np.linspace(oy-w/2, oy+w/2, num=10)
    z = np.linspace(oz-h/2, oz+h/2, num=10)
    x1, z1 = np.meshgrid(x, z)
    y11 = np.ones_like(x1)*(oy-w/2)
    y12 = np.ones_like(x1)*(oy+w/2)
    x2, y2 = np.meshgrid(x, y)
    z21 = np.ones_like(x2)*(oz-h/2)
    z22 = np.ones_like(x2)*(oz+h/2)
    y3, z3 = np.meshgrid(y, z)
    x31 = np.ones_like(y3)*(ox-l/2)
    x32 = np.ones_like(y3)*(ox+l/2)

    # outside surface
    ax.plot_surface(x1, y11, z1, color=color,  alpha=alpha)
    # inside surface
    ax.plot_surface(x1, y12, z1, color=color,  alpha=alpha)
    # bottom surface
    ax.plot_surface(x2, y2, z21, color=color,  alpha=alpha)
    # upper surface
    ax.plot_surface(x2, y2, z22, color=color,  alpha=alpha)
    # left surface
    ax.plot_surface(x31, y3, z3, color=color,  alpha=alpha)
    # right surface
    ax.plot_surface(x32, y3, z3, color=color, alpha=alpha)


def plot_inertia_cuboid(lxyz, pcom, ax):
    org = lxyz - 0.5 * pcom
    org = tuple(org.flatten().tolist())
    lxyz = tuple(lxyz.flatten().tolist())
    plot_cuboid(org, lxyz, ax, 'g', 0.1)




if __name__ == "__main__":
    CENTER = [0, 0, 0]
    LENGTH = 32 * 2
    WIDTH = 50 * 2
    HEIGHT = 100 * 2
    fig = plt.gcf()
    ax = Axes3D(fig)
    plot_cuboid(CENTER, (LENGTH, WIDTH, HEIGHT), ax, 'g', 1.0)
    plt.show()
