# %%
from io import IncrementalNewlineDecoder
import matplotlib
import numpy as np
import os
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
SCRIPTS_PATH = os.path.dirname(CURRENT_PATH)
MOTION_DATA_PATH = os.path.join(SCRIPTS_PATH, "motion_data")
MODEL_DATA_PATH = os.path.join(SCRIPTS_PATH, "model_data")
# %%
# Load motion data.
x = np.loadtxt(os.path.join(MOTION_DATA_PATH, "half_max.dat"))
q = x[0:7, :]
qdot = x[7:14, :]
x0 = x[:, 0]
q0 = q[:, 0]
qdot0 = qdot[:, 0]
# %%
from jbdl.rbdl.utils import ModelWrapper
mdlw = ModelWrapper()
mdlw.load(os.path.join(MODEL_DATA_PATH, 'half_max_v1.json'))
model = mdlw.model
# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from jbdl.rbdl.tools import plot_model
import math
%matplotlib inline
fig = plt.gcf()
ax = Axes3D(fig)
ax.set_ylim3d(-1.0, 1.0)
plot_model(model, q0, ax)
ax.view_init(elev=0,azim=-90)
ax.set_xlabel('X')
ax.set_xlim(-0.3, -0.3+0.8)
ax.set_ylabel('Y')
ax.set_ylim(-0.15, -0.15+0.8)
ax.set_zlabel('Z')
ax.set_zlim(-0.1, -0.1+0.8)
plt.show()
# %%
%matplotlib inline
plt.plot(q[2,:])

# %%
import matplotlib.pyplot as plt
%matplotlib 
fig = plt.figure()
ax = Axes3D(fig)
for i in range(q.shape[1]):
    print(i)
    ax.clear()
    plot_model(model, q[:, i], ax)
    ax.view_init(elev=0,azim=-90)
    ax.set_xlabel('X')
    ax.set_xlim(-0.3, -0.3+0.6)
    ax.set_ylabel('Y')
    ax.set_ylim(-0.15, -0.15+0.6)
    ax.set_zlabel('Z')
    ax.set_zlim(-0.1, -0.1+0.6)
    plt.pause(0.001)
    plt.show()
