import numpy as np


def clac_inertia_cuboid(inertia: np.ndarray, mass: float) -> np.ndarray:

    a = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ])

    b = np.divide(np.multiply(mass, inertia), 12.0)
    c = np.linalg.solve(a, b)
    lxyz = np.reshape(np.sqrt(c), (3, 1))
    return lxyz
