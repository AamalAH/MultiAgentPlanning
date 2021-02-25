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
    
def generate
    
def getActionProbs(Q, agentParams):
    """
        qValues: nPlayer x nActions x nSim
        return: nPlayer x nActions x nSim
        """
    alpha, tau, gamma = agentParams

    return np.exp(tau * Q) / np.sum(np.exp(tau * Q), axis=1)[:, None]

def getCurrentActions(actionProbs):
    return [np.random.choice([0, 1], p=actionProbs[p, :]) for p in range(2)]


def getRewards(G, bChoice):
    A, B = G

    rewards = np.zeros(2)
    rewards[0] = A[bChoice[0], bChoice[1]]
    rewards[1] = B[bChoice[0], bChoice[1]]

    return rewards

def qUpdate(Q, G, agentParams):

    alpha, tau, gamma = agentParams

    actionProbs = getActionProbs(Q, agentParams)
    bChoice = getCurrentActions(actionProbs)
    rewards = getRewards(G, bChoice)

    for p in range(2):
        Q[p, bChoice[p]] += alpha * (rewards[p] - Q[p, bChoice[p]] + gamma * np.max(Q[p, :]))
    return Q

def initialiseQ():
    return np.random.rand(2, 2)

def simulate(agentParams, nIter = 5e3):
    nIter = int(nIter)

    G = generateMatchingPennies()
    Q = initialiseQ()

    firstActionTracker = np.zeros((2, nIter))

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

    xdot[0] = alpha * x[0] * tau * ((A @ y)[0] - np.dot(x, A @ y)) + alpha * x[0] * (x[1] * np.log(x[1]/x[0]))
    xdot[1] = alpha * x[1] * tau * ((A @ y)[1] - np.dot(x, A @ y)) + alpha * x[1] * (x[0] * np.log(x[0]/x[1]))

    ydot[0] = alpha * y[0] * tau * ((B.T @ x)[0] - np.dot(y, B.T @ x)) + alpha * y[0] * (y[1] * np.log(y[1]/y[0]))
    ydot[1] = alpha * y[1] * tau * ((B.T @ x)[1] - np.dot(y, B.T @ x)) + alpha * y[1] * (y[0] * np.log(y[0]/y[1]))
   
    return np.hstack((xdot, ydot))

