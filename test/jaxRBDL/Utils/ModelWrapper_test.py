
import unittest
import os 
from oct2py import octave
from jaxRBDL.Utils.ModelWrapper import ModelWrapper
import numpy as np

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OCTAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(CURRENT_PATH)), "octave")
MRBDL_PATH = os.path.join(OCTAVE_PATH, "mRBDL")
MATH_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Math")
MODEL_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Model")
TOOLS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Tools")
KINEMATICS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Kinematics")
DYNAMICS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Dynamics")
IPPHYSICIALPARAMS_PATH = os.path.join(OCTAVE_PATH, "ipPhysicalParams") 
IPTOOLS_PATH = os.path.join(OCTAVE_PATH, "ipTools")
DATA_PATH  = os.path.join(os.path.dirname(CURRENT_PATH), "Data")

octave.addpath(MRBDL_PATH)
octave.addpath(MATH_PATH)
octave.addpath(MODEL_PATH)
octave.addpath(TOOLS_PATH)
octave.addpath(KINEMATICS_PATH)
octave.addpath(DYNAMICS_PATH)
octave.addpath(IPPHYSICIALPARAMS_PATH)
octave.addpath(IPTOOLS_PATH)
octave.addpath(OCTAVE_PATH)



class TestModelWrapper(unittest.TestCase):
    def setUp(self):
        ip = dict()
        model = dict()
        octave.push("ip", ip)
        octave.push("model", model)
        octave.push("ip", ip)
        octave.push("model", model)
        self.ip = octave.ipParmsInit(0, 0, 0, 0)
        self.model = octave.model_create()


    def test_ModelWrapper(self):
        mdl = ModelWrapper(self.model)

        self.assertIsInstance(mdl.parent, list)
        for item in mdl.parent:
            self.assertIsInstance(item, int)
        
        self.assertIsInstance(mdl.NB, int)
        self.assertIsInstance(mdl.a_grav, np.ndarray)
        self.assertTrue(mdl.a_grav.shape==(6, 1))

        self.assertIsInstance(mdl.jtype, list)
        for item in mdl.jtype:
            self.assertIsInstance(item, int)

        self.assertIsInstance(mdl.jaxis, str)
        for item in mdl.jaxis:
            self.assertIn(item, 'xyz')

        self.assertIsInstance(mdl.I, list)
        for item in mdl.I:
            self.assertIsInstance(item, np.ndarray)

        self.assertIsInstance(mdl.Xtree, list)
        for item in mdl.Xtree:
            self.assertIsInstance(item, np.ndarray)

        mdl.save(os.path.join(DATA_PATH, "MaxHalf.json"))
        mdl.load(os.path.join(DATA_PATH, "MaxHalf.json"))
        
        self.assertIsInstance(mdl.parent, list)
        for item in mdl.parent:
            self.assertIsInstance(item, int)
        
        self.assertIsInstance(mdl.NB, int)
        self.assertIsInstance(mdl.a_grav, np.ndarray)
        self.assertTrue(mdl.a_grav.shape==(6, 1))


        self.assertIsInstance(mdl.jtype, list)
        for item in mdl.jtype:
            self.assertIsInstance(item, int)

        self.assertIsInstance(mdl.jaxis, str)
        for item in mdl.jaxis:
            self.assertIn(item, 'xyz')

        self.assertIsInstance(mdl.I, list)
        for item in mdl.I:
            self.assertIsInstance(item, np.ndarray)

        self.assertIsInstance(mdl.Xtree, list)
        for item in mdl.Xtree:
            self.assertIsInstance(item, np.ndarray)

   



        





        
if __name__ == "__main__":
    unittest.main()