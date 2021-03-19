import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sps
from scipy.integrate import odeint

def generateMatchingPennies():
    """
    Create Matching Pennies Matrix

    :return:
    """

    C = np.array([[1, -1], [-1, 1]])
    Z = np.array([[2.4, -2], [-2, 2]])

    return C, Z


def replicatorODE(X, t, C, Z):

    x = X[0:2]
    y = X[2:4]
    z = X[4:]

    xdot = np.zeros(2)
    ydot = np.zeros(2)
    zdot = np.zeros(2)

    xdot[0] = x[0] * ((C @ y)[0] + (C @ z)[0] - np.dot(x, (C @ y) + (C @ z)))
    xdot[1] = x[1] * ((C @ y)[1] + (C @ z)[1] - np.dot(x, (C @ y) + (C @ z)))

    ydot[0] = y[0] * ((Z @ z)[0] + (C.T @ x)[0] - np.dot(y, (Z @ z) + (C.T @ x)))
    ydot[1] = y[1] * ((Z @ z)[1] + (C.T @ x)[1] - np.dot(y, (Z @ z) + (C.T @ x)))

    zdot[0] = z[0] * ((C.T @ x)[0] + ((-Z).T @ y)[0] - np.dot(z, (C.T @ x) + ((-Z).T @ y)))
    zdot[1] = z[1] * ((C.T @ x)[1] + ((-Z).T @ y)[1] - np.dot(z, (C.T @ x) + ((-Z).T @ y)))

    return np.hstack((xdot, ydot, zdot))


if __name__ == '__main__':

    nInit = 20

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

        x0 = np.random.rand(3)
        x0 = (np.vstack((x0, 1 - x0)).T).reshape(6)

        t = np.linspace(0, int(1e2), int(1e3) + 1)

        sol = odeint(replicatorODE, x0, t, args=G)
        ax.plot(sol[:, 0], sol[:, 2], sol[:, 4])
        ax.scatter(sol[0, 0], sol[0, 2], sol[0, 4], marker='o', color='r')

    plt.show()