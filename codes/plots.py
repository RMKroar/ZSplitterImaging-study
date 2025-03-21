import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

def plotResults(raw, EV, depthCode):
    depthCode = cp.asnumpy(depthCode)
    raw = cp.asnumpy(raw)   
    EV = cp.asnumpy(EV)

    edof = np.squeeze(np.sum(raw[..., np.newaxis] * depthCode, axis=2))
    edof = edof - np.min(edof)
    edof = edof / np.max(edof)

    plt.imshow(edof, cmap='gray')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.title('Raw EDOF Stack')
    plt.show()

    edof = np.squeeze(np.sum(EV[..., np.newaxis] * depthCode, axis=2))
    edof = edof - np.min(edof)
    edof = edof / np.max(edof)

    plt.imshow(edof, cmap='gray')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.title('EV-3D EDOF Stack')
    plt.show()
