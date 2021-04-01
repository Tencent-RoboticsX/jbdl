import matplotlib.pyplot as plt
import numpy as np
import os
from jaxRBDL.Utils.ModelWrapper import ModelWrapper
from mpl_toolkits.mplot3d.axes3d import Axes3D
from jaxRBDL.Tools.PlotModel import PlotModel

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
print(CURRENT_PATH)
mdlw = ModelWrapper()
mdlw.load(os.path.join(CURRENT_PATH, 'whole_max_v1.json'))
print(type(mdlw))
model = mdlw.model

q = np.array([0, 0, 0.27, 0, 0, 0,
    0, 0.5, -0.8,  # fr
    0, 0.5, -0.8,  # fl
    0, 0.5, -0.8,  # br
    0, 0.5, -0.8]) # bl






fig = plt.gcf()
ax = Axes3D(fig)
ax.set_ylim3d(-1.0, 1.0)
PlotModel(model, q, ax)

ax.view_init(elev=0,azim=-90)
ax.set_xlabel('X')
ax.set_xlim(-0.3, -0.3+0.8)
ax.set_ylabel('Y')
ax.set_ylim(-0.15, -0.15+0.8)
ax.set_zlabel('Z')
ax.set_zlim(-0.1, -0.1+0.8)


plt.show()

plt.ion()
plt.figure()
fig = plt.gcf()
ax = Axes3D(fig)  
for i in range(1000):
    print(i)

    q0 = np.array([0, 0, 0.5 - i * 0.001, 0.0, 0, 0,
        0, 0.5, -0.8,  # fr
        0, 0.5, -0.8,  # fl
        0, 0.5, -0.8,  # br
        0, 0.5, -0.8]) # bl

    ax = plt.gca()
    ax.clear()
    PlotModel(model, q0, ax)
    ax.view_init(elev=0,azim=-90)
    ax.set_xlabel('X')
    ax.set_xlim(-0.3, -0.3+0.6)
    ax.set_ylabel('Y')
    ax.set_ylim(-0.15, -0.15+0.6)
    ax.set_zlabel('Z')
    ax.set_zlim(-0.1, -0.1+0.6)
    plt.pause(0.001)
    plt.show()

plt.ioff()


