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

    C = np.array([[5, -5], [-5, 5]])
    Z = np.array([[1, -1], [-1, 1]])

    return C, Z

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

    C, Z = G

    alpha, tau, gamma = agentParams

    x = X[0:2]
    y = X[2:4]
    z = X[4:]

    xdot = np.zeros(2)
    ydot = np.zeros(2)
    zdot = np.zeros(2)

    xdot[0] = alpha * x[0] * tau * ((Z @ y)[0] + (Z @ z)[0] - np.dot(x, (Z @ y) + (Z @ z))) + alpha * x[0] * (x[1] * np.log(x[1]/x[0]))
    xdot[1] = alpha * x[1] * tau * ((Z @ y)[1] + (Z @ z)[1] - np.dot(x, (Z @ y) + (Z @ z))) + alpha * x[1] * (x[0] * np.log(x[0]/x[1]))

    ydot[0] = alpha * y[0] * tau * ((C @ z)[0] + ((-Z).T @ x)[0] - np.dot(y, (C @ z) + ((-Z).T @ x))) + alpha * y[0] * (y[1] * np.log(y[1]/y[0]))
    ydot[1] = alpha * y[1] * tau * ((C @ z)[1] + ((-Z).T @ x)[1] - np.dot(y, (C @ z) + ((-Z).T @ x))) + alpha * y[1] * (y[0] * np.log(y[0]/y[1]))

    zdot[0] = alpha * z[0] * tau * (((-Z) @ x)[0] + (C.T @ y)[0] - np.dot(z, ((-Z) @ x) + (C.T @ y))) + alpha * z[0] * (z[1] * np.log(z[1]/z[0]))
    zdot[1] = alpha * z[1] * tau * (((-Z) @ x)[1] + (C.T @ y)[1] - np.dot(z, ((-Z) @ x) + (C.T @ y))) + alpha * z[1] * (z[0] * np.log(z[0]/z[1]))

    return np.hstack((xdot, ydot, zdot))


if __name__ == '__main__':

    nInit = 20

    G = generateMatchingPennies()
    alpha, tau, gamma = 0.1, 10, 0.1
    agentParams = alpha, tau, gamma

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('P1')
    ax.set_ylabel('P2')
    ax.set_zlabel('P3')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    for cInit in range(nInit):
        Q = initialiseQ()

        alpha, tau, gamma = 0.1, 20, 0.1
        agentParams = alpha, tau, gamma

        x0 = getActionProbs(Q, agentParams).reshape(6)

        alpha, tau, gamma = 1, 1, 0.1
        agentParams = alpha, tau, gamma

        t = np.linspace(0, int(1e4), int(1e5) + 1)

        sol = odeint(TuylsODE, x0, t, args=(G, agentParams))
        ax.plot(sol[:, 0], sol[:, 2], sol[:, 4])
        ax.scatter(sol[0, 0], sol[0, 2], sol[0, 4], marker='o', color='r')

    plt.title('alpha = {0}, tau = {1}'.format(np.round(alpha, 2), np.round(tau, 2)))

    plt.show()