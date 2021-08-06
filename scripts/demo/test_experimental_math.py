from jbdl.experimental.math import subtract, compute_eccentric_anomaly
from jbdl.experimental import math
from jbdl.experimental import tools
from jbdl.experimental.tools import multiply
from jbdl.experimental.tools import Pet
from jbdl.experimental import qpoases
from jbdl.experimental.qpoases import QProblem
# from jbdl.experimental.cpu_ops import lcp
#from jbdl.experimental.custom_ops.lcp import lcp
# from jbdl.experimental.custom_ops.lcp_gpu import lcp_gpu
import numpy as np
from jax.lib import xla_bridge as xb
if xb.get_backend().platform == 'gpu':
    from jbdl.experimental.custom_ops.lcp_gpu import lcp_gpu as lcp
else:
    from jbdl.experimental.custom_ops.lcp import lcp

print(math.__name__)
print(tools.__name__)
print(qpoases.__name__)
print(subtract(10, -2))
print(multiply(10, 2))
my_dog = Pet('Pluto', 5)
print(my_dog.get_name())
print(my_dog.get_hunger())
my_dog.go_for_a_walk()
print(my_dog.get_hunger())
qp_problem = QProblem(2, 1)
print(qp_problem.init([1.0, 0.0, 0.0, 0.5], [1.0, 1.0], [1.5, 1.0], [0.5, -2.0], [5.0, 2.0 ], [-1.0 ], [2.0], 10))
print(qp_problem.getPrimalSolution())
print(qp_problem.getDualSolution())
print(qp_problem.getObjVal())
print(compute_eccentric_anomaly(10.0, 20.0))
# print(lcp([1.0, -1.0, -1.0, 2.0],  [-2.0, -6.0],  [1.0, 1.0, -1.0, 2.0, 2.0, 1.0], [2.0, 2.0, 3.0], [0.0, 0.0], [0.5, 5.0 ], 2, 3))

H = np.array([1.0, -1.0, -1.0, 2.0]).reshape(2, 2)
f = np.array([-2.0, -6.0]).reshape(2, 1)
L = np.array([1.0, 1.0, -1.0, 2.0, 2.0, 1.0]).reshape(3, 2)
k = np.array([2.0, 2.0, 3.0]).reshape(3, 1)
lb = np.array([0.0, 0.0]).reshape(2, 1)
ub = np.array([0.5, 5.0 ]).reshape(2, 1)

print(lcp(H, f, L, k, lb, ub))