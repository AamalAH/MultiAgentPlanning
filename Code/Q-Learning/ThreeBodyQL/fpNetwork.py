import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sps
from scipy.integrate import odeint
from tqdm import tqdm

def generateMatchingPennies():
    """
    Create Matching Pennies Matrix

    :return:
    """

    C = np.array([[1, -1], [-1, 1]])
    Z = np.array([[2.4, -2], [-2, 2]])

    return C, Z

def initialiseRandomVec():
    return np.random.dirichlet(np.ones(2), size=(3)).reshape(6)

def FPOrbit(x0, G, nIter):
    Z, C = G

    x, y, z = x0[0:2], x0[2:4], x0[4:]

    allX = np.zeros((nIter, 6))

    for n in tqdm(range(1, nIter + 1)):
        eX, eY, eZ = np.zeros(2), np.zeros(2), np.zeros(2)
        brX = np.argmax(C @ y + C @ z)
        brY = np.argmax(C.T @ x + Z @ z)
        brZ = np.argmax((C.T @ x + (-Z).T @ y))

        eX[brX] = 1
        eY[brY] = 1
        eZ[brZ] = 1

        x = (n * x + eX) / (n + 1)
        y = (n * y + eY) / (n + 1)
        z = (n * z + eZ) / (n + 1)

        allX[n-1] = np.hstack((x, y, z))

    return allX

if __name__ == "__main__":
    nInit = 5
    nIter = 1e5

    G = generateMatchingPennies()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('P1')
    ax.set_ylabel('P2')
    ax.set_zlabel('P3')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    for cInit in range(nInit):

        x0 = initialiseRandomVec()
        sol = FPOrbit(x0, G, int(nIter))

        ax.plot(sol[:, 0], sol[:, 2], sol[:, 4])
        ax.scatter(sol[0, 0], sol[0, 2], sol[0, 4], marker='o', color='r')

    plt.show()