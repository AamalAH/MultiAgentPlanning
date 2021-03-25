import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sps
from scipy.integrate import odeint

def generateMixedTests():
    """
    Create Matching Pennies Matrix

    :return:
    """

    A = np.array([[1, -1], [-1, 1]])
    B = np.array([[-1, 1], [1, -1]])
    # C = np.copy([[0.2, -0.2], [-0.2, 0.2]])
    return A, B, C

def replicatorzcc(X, t, A, B, C):

    x = X[0:2]
    y = X[2:4]
    z = X[4:]

    xdot = np.zeros(2)
    ydot = np.zeros(2)
    zdot = np.zeros(2)

    xdot[0] = x[0] * ((A @ y)[0] + (B.T @ z)[0] - np.dot(x, (A @ y) + (B.T @ z)))
    xdot[1] = x[1] * ((A @ y)[1] + (B.T @ z)[1] - np.dot(x, (A @ y) + (B.T @ z)))

    ydot[0] = y[0] * ((B @ z)[0] + (B.T @ x)[0] - np.dot(y, (5*A @ z) + (B.T @ x)))
    ydot[1] = y[1] * ((5*A @ z)[1] + (B.T @ x)[1] - np.dot(y, (5*A @ z) + (B.T @ x)))

    zdot[0] = z[0] * ((A @ x)[0] + (5*B.T @ y)[0] - np.dot(z, (A @ x) + (5*B.T @ y)))
    zdot[1] = z[1] * ((A @ x)[1] + (5*B.T @ y)[1] - np.dot(z, (A @ x) + (5*B.T @ y)))

    return np.hstack((xdot, ydot, zdot))