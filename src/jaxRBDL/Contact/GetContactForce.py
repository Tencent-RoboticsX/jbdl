import numpy as np

def GetContactForce(model: dict, fqp: np.ndarray, fpd: np.ndarray, flag_contact: np.ndarray):
    fqp = fqp.flatten()
    fpd = fpd.flatten()

    NC = int(model["NC"])
    nf = int(model["nf"])
    fc = np.zeros((3*NC,))
    fcqp = np.zeros((3*NC,))
    fcpd = np.zeros((3*NC,))
    k = 0
    for i in range(NC):
        if flag_contact[i]!=0:
            if nf==2: # Only x/z direction
                fc[3*i:3*i+3] = np.array([fqp[k*nf] + fpd[k*nf], 0.0, fqp[k*nf+nf-1] + fpd[k*nf+nf-1]])
                fcqp[3*i:3*i+3] = np.array([fqp[k*nf], 0.0, fqp[k*nf+nf-1]])
                fcpd[3*i:3*i+3] = np.array([fpd[k*nf], 0.0, fpd[k*nf+nf-1]])
            else: 
                fc[3*i:3*i+3] = fqp[k*nf:k*nf+nf] + fpd[k*nf:k*nf+nf] 
                fcqp[3*i:3*i+3] = fqp[k*nf:k*nf+nf]
                fcpd[3*i:3*i+3] = fpd[k*nf:k*nf+nf]
            k = k+1

    fc = fc.reshape(-1, 1)
    fcqp = fcqp.reshape(-1, 1)
    fcpd = fcpd.reshape(-1, 1)
    return fc, fcqp, fcpd

