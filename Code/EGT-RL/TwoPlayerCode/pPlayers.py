import numpy as np
import itertools
import scipy.sparse as sps
import matplotlib.pyplot as plt
from tqdm import tqdm

t0 = 500
gamma = .1
tau = 5e-2

initnSim = 10
nIter = int(5e4)
nTests = 10

nPlayers = 2
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

    allPayoffs = np.array(([1, 5, 0, 3], [1, 5, 0, 3]))

    for i in range(nSim):
        rewards = np.random.multivariate_normal(np.zeros(nPlayers * nElements), cov=cov)
        each = np.array_split(rewards, nPlayers)
        allPayoffs = np.dstack((allPayoffs, np.array(each)))

    return allPayoffs[:, :, 1:]

def getActionProbs(qValues):
    partitionFunction = np.expand_dims(np.sum(np.exp(tau * qValues), axis=1), axis = 1)
    partitionFunction = np.hstack([partitionFunction]*nActions)
    actionProbs = np.exp(tau * qValues)/partitionFunction
    return actionProbs


def qUpdate(qValues, payoffs, nSim):
    idX = list(itertools.product(list(range(0, nActions)), repeat=nPlayers))
    actionProbs = getActionProbs(qValues)
    bChoice = np.array([[np.random.choice(list(range(nActions)), p=actionProbs[p, :, s]) for s in range(nSim)] for p in range(nPlayers)])

    rewards = np.array([[payoffs[p, np.where([np.all(idX[i] == np.roll(bChoice[:, s], -1 * p)) for i in range(len(idX))])[0][0], s] for s in range(nSim)] for p in range(nPlayers)])
    for s in range(nSim):
        qValues[range(nPlayers), bChoice[:, s], s] += alpha * (rewards[:, s] - qValues[range(nPlayers), bChoice[:, s], s] + gamma * np.max(qValues[:, :, s], axis = 1))

    return qValues

def checkminMax(allActions, nSim, tol):
    a = np.reshape(allActions, (t0, nSim, nActions * 2))
    relDiff = ((np.max(a, axis=0) - np.min(a, axis=0))/np.min(a, axis=0))
    return np.all(relDiff < tol, axis=1)

plotConv = []

for alpha in tqdm(np.linspace(1e-2, 5e-2, num=10)):
    for Gamma in np.linspace(-1, 0, num=10):

        allActions = []
        nSim = initnSim
        converged = 0

        payoffs = generateGames(Gamma, nSim, nActions, nPlayers)
        qValues0 = np.random.rand(nPlayers, nActions, nSim)

        for cIter in range(nIter):

            if cIter == t0:
                allActions = []

            if cIter%t0 == 0 and cIter != 0 and cIter != t0:

                vars = checkminMax(np.array(allActions), nSim, 1e-2)
                idx = np.where(vars)
                qValues0 = np.delete(qValues0, idx, axis=2)
                nSim -= len(idx[0])
                converged += len(idx[0])
                allActions = []

            if nSim <= 0:
                break

            allActions += [getActionProbs(qValues0)]
            qValues0 = qUpdate(qValues0, payoffs, nSim)

        plotConv += [np.array([alpha, Gamma, converged / initnSim])]