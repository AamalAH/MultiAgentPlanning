import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

alpha = .1
gamma = .75
Gamma = 0.1
tau = 2

delta_0 = 1e-3
nSim = 10
nIter = int(1e4)


def generateGames(gamma, nSim, nAct):
    """
    Draw a random payoff matrix from a multivariate Gaussian, currently the unscaled version.

    gamma: Choice of co-operation parameter
    nAct: Number of actions in the game

    [reward1s, reward2s]: list of payoff matrices
    """

    nElements = nAct ** 2  # number of payoff elements in the matrix

    cov = np.eye(2 * nElements)  # <a_ij^2> = <b_ji^2> = 1
    cov[:nElements, nElements:] = np.eye(nElements) * gamma  # <a_ij b_ji> = Gamma
    cov[nElements:, :nElements] = np.eye(nElements) * gamma

    rewardAs, rewardBs = np.eye(2), np.eye(2)

    for i in range(nSim):

        rewards = np.random.multivariate_normal(np.zeros(2 * nElements), cov=cov)

        # rewardAs = np.dstack((rewardAs, rewards[0:nElements].reshape((nAct, nAct))))
        rewardAs = np.dstack((rewardAs, np.array([[1, 0], [0, 2]])))
        # rewardBs = np.dstack((rewardBs, rewards[nElements:].reshape((nAct, nAct)).T))
        rewardBs = np.dstack((rewardBs, np.array([[2, 0], [0, 1]])))
    return [rewardAs[:, :, 1:], rewardBs[:, :, 1:]]

def getActionProbs(qValues, nSim):
    partitionFunction = np.sum(np.exp(tau * qValues), axis = 1)
    actionProbs = np.array([np.array([np.exp(tau * qValues[p, :, s])/partitionFunction[p, s] for p in range(2)]) for s in range(nSim)])

    return actionProbs

def qUpdate(qValues, payoffs):
    actionProbs = getActionProbs(qValues, nSim)

    boltzmannChoices = np.array([[np.random.choice([0, 1], p=actionProbs[s, p, :]) for p in range(2)] for s in range(nSim)])

    rewardAs = payoffs[0][boltzmannChoices[:, 0], boltzmannChoices[:, 1], (range(nSim))]
    rewardBs = payoffs[1][boltzmannChoices[:, 0], boltzmannChoices[:, 1], (range(nSim))]

    qValues[[0]*nSim, boltzmannChoices[:, 0], (range(nSim))] += alpha * (
                rewardAs - qValues[[0]*nSim, boltzmannChoices[:, 0], (range(nSim))] + gamma * np.max(qValues[[0]*nSim, :, (range(nSim))], axis=1))

    qValues[[1]*nSim, boltzmannChoices[:, 1], list(range(nSim))] += alpha * (
                rewardBs - qValues[[1]*nSim, boltzmannChoices[:, 1], (range(nSim))] + gamma * np.max(qValues[[1]*nSim, :, (range(nSim))], axis=1))

    return qValues

def getDelta(qValues0, qValues1, nSim):

    actionProbs0, actionProbs1 = getActionProbs(qValues0, nSim), getActionProbs(qValues1, nSim)
    return np.mean(abs(actionProbs1 - actionProbs0), axis=0)

allExpo = np.zeros((10, 10))

i = 0
for alpha in tqdm(np.linspace(0.1, 1, num=10)):
    j = 1
    for Gamma in np.linspace(-1, 1, num=10):

        all_actionProbs = np.eye(2)

        payoffs = generateGames(Gamma, nSim, 2)

        qValues0 = np.random.rand(2, 2, nSim)
        qValues1 = np.random.rand(2, 2, nSim)
        # qValues1 = qValues0 + np.array([np.random.choice([-delta_0, delta_0]) for i in range(nSim * 4)]).reshape((2, 2, nSim))

        for cIter in range(nIter):
            for i in range(nSim):
                all_actionProbs = np.dstack((all_actionProbs, (getActionProbs(qValues0, nSim)[i, :, :])))
            if cIter % 300 == 0 and cIter != 0:
                delta_1 = getDelta(qValues0, qValues1, nSim)
            qValues0, qValues1 = qUpdate(qValues0, payoffs), qUpdate(qValues1, payoffs)

        delta_n = getDelta(qValues0, qValues1, nSim)

        liapExp = np.max((1 / nIter) * np.log(delta_n / delta_1))

        allExpo[10 - j, i] = liapExp

        j += 1
    i += 1