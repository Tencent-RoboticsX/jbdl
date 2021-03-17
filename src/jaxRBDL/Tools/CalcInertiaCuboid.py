import numpy as np

def ClacInertiaCuboid(Inertia: np.ndarray, mass: float)-> np.ndarray:

    A = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ])

    B = np.divide(np.multiply(mass, Inertia), 12.0)
    C = np.linalg.solve(A, B)
    lxyz = np.reshape(np.sqrt(C), (3, 1))
    return lxyz

