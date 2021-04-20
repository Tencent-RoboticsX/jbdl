# %%
import numpy as np
import os
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MOTION_DATA_PATH = os.path.join(CURRENT_PATH, "motion_data")
# %%
# Load motion data.
x = np.loadtxt(os.path.join(MOTION_DATA_PATH, "half_max.dat"))
q = x[:, 0:7]
qdot = x[:, 7:14]
x0 = x[:, 0]
q0 = q[:, 0]
qdot0 = qdot[:, 0]
# %%
from jbdl.rbdl.utils import ModelWrapper