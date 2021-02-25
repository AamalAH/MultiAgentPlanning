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

def getActionProbs(Q, agentParams):
    """
        qValues: nPlayer x nActions x nSim
        return: nPlayer x nActions x nSim
        """
    alpha, tau, gamma = agentParams

    return np.exp(tau * Q) / np.sum(np.exp(tau * Q), axis=1)[:, None]

def getCurrentActions(actionProbs):
    return [np.random.choice([0, 1], p=actionProbs[p, :]) for p in range(3)]

def getRewards(G, bChoice):
    A, B = G

    rewards = np.zeros(3)
    rewards[0] = A[bChoice[0], bChoice[1]] + B[bChoice[2], bChoice[0]]
    rewards[1] = B[bChoice[0], bChoice[1]] + A[bChoice[1], bChoice[2]]
    rewards[2] = A[bChoice[2], bChoice[0]] + B[bChoice[1], bChoice[2]]

    return rewards

def qUpdate(Q, G, agentParams):

    alpha, tau, gamma = agentParams

    actionProbs = getActionProbs(Q, agentParams)
    bChoice = getCurrentActions(actionProbs)
    rewards = getRewards(G, bChoice)

    for p in range(3):
        Q[p, bChoice[p]] += alpha * (rewards[p] - Q[p, bChoice[p]] + gamma * np.max(Q[p, :]))
    return Q

def initialiseQ():
    return np.random.rand(3, 2)

def simulate(agentParams, nIter = 5e3):
    nIter = int(nIter)

    G = generateMatchingPennies()
    Q = initialiseQ()

    firstActionTracker = np.zeros((3, nIter))

    for cIter in range(nIter):
        Q = qUpdate(Q, G, agentParams)
        firstActionTracker[:, cIter] = getActionProbs(Q, agentParams)[:, 0]

    return firstActionTracker

def TuylsODE(X, t, G, agentParams):

    A, B = G

    alpha, tau, gamma = agentParams

    x = X[0:2]
    y = X[2:4]
    z = X[4:]

    xdot = np.zeros(2)
    ydot = np.zeros(2)
    zdot = np.zeros(2)

    xdot[0] = alpha * x[0] * tau * ((A @ y)[0] + (z.T @ B)[0] - np.dot(x, (A @ y) + (z.T @ B))) + alpha * x[0] * (x[1] * np.log(x[1]/x[0]))
    xdot[1] = alpha * x[1] * tau * ((A @ y)[1] + (z.T @ B)[1] - np.dot(x, (A @ y) + (z.T @ B))) + alpha * x[1] * (x[0] * np.log(x[0]/x[1]))

    ydot[0] = alpha * y[0] * tau * ((A @ z)[0] + (x.T @ B)[0] - np.dot(y, (A @ z) + (x.T @ B))) + alpha * y[0] * (y[1] * np.log(y[1]/y[0]))
    ydot[1] = alpha * y[1] * tau * ((A @ z)[1] + (x.T @ B)[1] - np.dot(y, (A @ z) + (x.T @ B))) + alpha * y[1] * (y[0] * np.log(y[0]/y[1]))

    zdot[0] = alpha * z[0] * tau * ((A @ x)[0] + (y.T @ B)[0] - np.dot(z, (A @ x) + (y.T @ B))) + + alpha * z[0] * (z[1] * np.log(z[1]/z[0]))
    zdot[1] = alpha * z[1] * tau * ((A @ x)[1] + (y.T @ B)[1] - np.dot(z, (A @ x) + (y.T @ B))) + alpha * z[1] * (z[0] * np.log(z[0]/z[1]))

    return np.hstack((xdot, ydot, zdot))


if __name__ == '__main__':

    Q = initialiseQ()
    G = generateMatchingPennies()

    alpha, tau, gamma = 0.05, 0.1, 0.1
    agentParams = alpha, tau, gamma

    x0 = getActionProbs(Q, agentParams).reshape(6)

    t = np.linspace(0, 10, 101)

    sol = odeint(TuylsODE, x0, t, args=(G, agentParams))

    # firstActionTracker = simulate(agentParams)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.plot(firstActionTracker[0, ::25], firstActionTracker[1, ::25], firstActionTracker[2, ::25]), plt.show()