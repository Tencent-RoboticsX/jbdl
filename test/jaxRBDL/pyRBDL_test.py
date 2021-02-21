import os
from oct2py import octave
import numpy as np
import math

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OCTAVE_PATH = os.path.join(os.path.dirname(CURRENT_PATH), "octave")
MRBDL_PATH = os.path.join(OCTAVE_PATH, "mRBDL")
MATH_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Math")
MODEL_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Model")
TOOLS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Tools")
KINEMATICS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Kinematics")
IPPHYSICIALPARAMS_PATH = os.path.join(OCTAVE_PATH, "ipPhysicalParams") 
IPTOOLS_PATH = os.path.join(OCTAVE_PATH, "ipTools")


octave.addpath(MRBDL_PATH)
octave.addpath(MATH_PATH)
octave.addpath(MODEL_PATH)
octave.addpath(TOOLS_PATH)
octave.addpath(KINEMATICS_PATH)
octave.addpath(IPPHYSICIALPARAMS_PATH)
octave.addpath(IPTOOLS_PATH)
octave.addpath(OCTAVE_PATH)

ip = dict()
model = dict()

octave.push("ip", ip)
octave.push("model", model)
ip = octave.ipParmsInit(0, 0, 0, 0)
model = octave.model_create()
# print(model["jtype"])
# print(model["jaxis"][0])
# print(model["Xtree"][0][0])
print(model["parent"])
# print(model["I"])

# q = np.array([0.0,  0.4765, 0, math.pi/6, math.pi/6, -math.pi/3, -math.pi/3])
# octave.PlotModel(model, q)
# octave.PlotCoMInertia(model, q)





