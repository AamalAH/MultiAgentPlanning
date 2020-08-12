import numpy as np
import itertools
import scipy.sparse as sps
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm


alpha = .1
gamma = .1
Gamma = 0.1
tau = 0.05

delta_0 = 1e-3
nSim = 5
nIter = int(1e4)

nPlayers = 3
nActions = 2

def generateGames(gamma, nSim, nAct, nPlayers):
    """
    Draw a random payoff matrix from a multivariate Gaussian, currently the unscaled version.

    gamma: Choice of co-operation parameter
    nAct: Number of actions in the game

    [reward1s, reward2s]: list of payoff matrices
    """

    nElements = nAct ** nPlayers  # number of payoff elements in a player's matrix

    cov = sps.lil_matrix(sps.eye(nPlayers * nElements))
    idX = list(itertools.product(list(range(1, nAct + 1)), repeat=nPlayers)) * nPlayers
    for idx in range(nElements):
        for b in range(1, nPlayers):
            rollPlayers = np.where([np.all(idX[i] == np.roll(idX[idx], -1 * b)) for i in range(len(idX))])[0][b]
            cov[idx, rollPlayers], cov[rollPlayers, idx] = gamma/(nPlayers - 1), gamma/(nPlayers - 1)

    cov = cov.toarray()

    allPayoffs = np.zeros(shape=(nPlayers, nElements))

    for i in range(nSim):
        rewards = np.random.multivariate_normal(np.zeros(nPlayers * nElements), cov=cov)
        each = np.array_split(rewards, nPlayers)
        allPayoffs = np.dstack((allPayoffs, np.array(each)))

    return allPayoffs


def getActionProbs(qValues):
    partitionFunction = np.expand_dims(np.sum(np.exp(tau * qValues), axis=1), axis = 1)
    partitionFunction = np.hstack([partitionFunction]*nActions)
    actionProbs = np.exp(tau * qValues)/partitionFunction
    return actionProbs


def qUpdate(qValues, payoffs):
    idX = list(itertools.product(list(range(0, nActions)), repeat=nPlayers))
    actionProbs = getActionProbs(qValues)
    bChoice = np.array([[np.random.choice([0, 1], p=actionProbs[p, :, s]) for s in range(nSim)] for p in range(nPlayers)])

    rewards = np.array([[payoffs[p, np.where([np.all(idX[i] == np.roll(bChoice[:, s], -1 * p)) for i in range(len(idX))])[0][0], s] for s in range(nSim)] for p in range(nPlayers)])
    for s in range(nSim):
        qValues[range(nPlayers), bChoice[:, s], s] += alpha * (rewards[:, s] - qValues[range(3), bChoice[:, s], s] + gamma * np.max(qValues[:, :, s], axis = 1))

    return qValues


def getDelta(qValues0, qValues1):
    actionProbs0, actionProbs1 = getActionProbs(qValues0), getActionProbs(qValues1)
    return np.mean(abs(actionProbs1 - actionProbs0), axis=2)


# allExpo = np.zeros((10, 10, 10))
plotExpo = []
i = 0
for alpha in tqdm(np.linspace(0, 1, num=10)):
    j = 1
    for tau in np.linspace(1, 10, num=10):
        k = 0
        for Gamma in np.linspace(-1, 1, num=10):

            all_actionProbs = np.eye(2)

            payoffs = generateGames(Gamma, nSim, nActions, nPlayers)

            qValues0 = np.random.rand(nPlayers, nActions, nSim)
            qValues1 = np.random.rand(nPlayers, nActions, nSim)

            for cIter in range(nIter):
                # for i in range(nSim):
                #     all_actionProbs = np.dstack((all_actionProbs, (getActionProbs(qValues0, nSim)[i, :, :])))
                if cIter % 300 == 0 and cIter != 0:
                    delta_1 = getDelta(qValues0, qValues1)
                qValues0, qValues1 = qUpdate(qValues0, payoffs), qUpdate(qValues1, payoffs)

            delta_n = getDelta(qValues0, qValues1)

            liapExp = np.max((1 / nIter) * np.log(delta_n / delta_1))

            plotExpo += [np.array([alpha, tau, Gamma, liapExp])]

            # allExpo[10 - j, i, k] = liapExp
            k += 1
        j += 1
    i += 1