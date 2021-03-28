from jaxRBDL.Utils.ModelWrapper import ModelWrapper
import os
import numpy as np


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
print(CURRENT_PATH)
mdlw = ModelWrapper()
mdlw.load(os.path.join(CURRENT_PATH, 'whole_max_v0.json'))


mdlw.contact_force_lb = np.array([-1000.0, -1000.0, 0.0]).reshape(-1, 1)
mdlw.contact_force_ub = np.array([1000.0, 1000.0, 3000.0]).reshape(-1, 1)
mdlw.contact_force_kp = np.array([10000.0, 10000.0, 10000.0]).reshape(-1, 1)
mdlw.contact_force_kd = np.array([1000.0, 1000.0, 1000.0]).reshape(-1, 1)

mdlw.contact_pos_lb = np.array([0.0001, 0.0001, 0.0001]).reshape(-1, 1)
mdlw.contact_pos_ub = np.array([0.0001, 0.0001, 0.0001]).reshape(-1, 1)
mdlw.contact_vel_lb = np.array([-0.05, -0.05, -0.05]).reshape(-1, 1)
mdlw.contact_vel_ub = np.array([0.01, 0.01, 0.01]).reshape(-1, 1)

mdlw.save(os.path.join(CURRENT_PATH, 'whole_max_v1.json'))