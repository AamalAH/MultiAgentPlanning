import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

if __name__ == '__main__':

    alpha, tau, gamma = 0.05, 0.1, 0.1

    agentParams = alpha, tau, gamma

    firstActionTracker = simulate(agentParams)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(firstActionTracker[0, ::25], firstActionTracker[1, ::25], firstActionTracker[2, ::25]), plt.show()