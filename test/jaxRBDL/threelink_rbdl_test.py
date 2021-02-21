
import os
from oct2py import octave
import numpy as np
import math

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OCTAVE_PATH = os.path.join(os.path.dirname(CURRENT_PATH), "octave")
octave.run(os.path.join(OCTAVE_PATH, 'threelink_rbdl_test.m'))