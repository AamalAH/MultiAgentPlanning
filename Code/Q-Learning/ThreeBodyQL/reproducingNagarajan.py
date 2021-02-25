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

    A = np.array([[1, -1], [-1, 1]])
    B = np.array([[-1, 1], [1, -1]])

    return A, B

def replicatorODE(X, t, A, B):

    x = X[0:2]
    y = X[2:4]
    z = X[4:]

    xdot = np.zeros(2)
    ydot = np.zeros(2)
    zdot = np.zeros(2)

    xdot[0] = x[0] * ((A @ y)[0] + (B.T @ z)[0] - np.dot(x, (A @ y) + (B.T @ z)))
    xdot[1] = x[1] * ((A @ y)[1] + (B.T @ z)[1] - np.dot(x, (A @ y) + (B.T @ z)))

    ydot[0] = y[0] * ((A @ z)[0] + (B.T @ x)[0] - np.dot(y, (A @ z) + (B.T @ x)))
    ydot[1] = y[1] * ((A @ z)[1] + (B.T @ x)[1] - np.dot(y, (A @ z) + (B.T @ x)))

    zdot[0] = z[0] * ((A @ x)[0] + (B.T @ y)[0] - np.dot(z, (A @ x) + (B.T @ y)))
    zdot[1] = z[1] * ((A @ x)[1] + (B.T @ y)[1] - np.dot(z, (A @ x) + (B.T @ y)))

    return np.hstack((xdot, ydot, zdot))


if __name__ == '__main__':

    G = generateMatchingPennies()

    x0 = np.random.rand(3)
    x0 = (np.vstack((x0, 1 - x0)).T).reshape(6)

    t = np.linspace(0, int(1e3), int(1e4) + 1)

    sol = odeint(replicatorODE, x0, t, args=G)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('P1')
    ax.set_ylabel('P2')
    ax.set_zlabel('P3')

    ax.plot(sol[:, 0], sol[:, 2], sol[:, 4]), plt.show()
