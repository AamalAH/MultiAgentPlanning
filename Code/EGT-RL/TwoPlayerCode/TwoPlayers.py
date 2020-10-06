import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

# alpha = .1
gamma = .1
Gamma = 0.1
tau = 5e-2

nActions = 5
t0 = 500

initnSim = 15

delta_0 = 1e-3
nIter = int(5e4)


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

    rewardAs, rewardBs = np.eye(nAct), np.eye(nAct)

    for i in range(nSim):
        rewards = np.random.multivariate_normal(np.zeros(2 * nElements), cov=cov)

        rewardAs = np.dstack((rewardAs, rewards[0:nElements].reshape((nAct, nAct))))
        rewardBs = np.dstack((rewardBs, rewards[nElements:].reshape((nAct, nAct)).T))

    return [rewardAs[:, :, 1:], rewardBs[:, :, 1:]]


def getActionProbs(qValues, nSim):
    partitionFunction = np.sum(np.exp(tau * qValues), axis=1)
    actionProbs = np.array(
        [np.array([np.exp(tau * qValues[p, :, s]) / partitionFunction[p, s] for p in range(2)]) for s in range(nSim)])

    return actionProbs


def qUpdate(qValues, payoffs, nSim):
    actionProbs = getActionProbs(qValues, nSim)

    boltzmannChoices = np.array(
        [[np.random.choice(list(range(nActions)), p=actionProbs[s, p, :]) for p in range(2)] for s in range(nSim)])

    rewardAs = payoffs[0][boltzmannChoices[:, 0], boltzmannChoices[:, 1], (range(nSim))]
    rewardBs = payoffs[1][boltzmannChoices[:, 0], boltzmannChoices[:, 1], (range(nSim))]

    qValues[[0] * nSim, boltzmannChoices[:, 0], (range(nSim))] += alpha * (
            rewardAs - qValues[[0] * nSim, boltzmannChoices[:, 0], (range(nSim))] + gamma * np.max(
        qValues[[0] * nSim, :, (range(nSim))], axis=1))

    qValues[[1] * nSim, boltzmannChoices[:, 1], list(range(nSim))] += alpha * (
            rewardBs - qValues[[1] * nSim, boltzmannChoices[:, 1], (range(nSim))] + gamma * np.max(
        qValues[[1] * nSim, :, (range(nSim))], axis=1))

    return qValues


def getDelta(qValues0, qValues1, nSim):
    actionProbs0, actionProbs1 = getActionProbs(qValues0, nSim), getActionProbs(qValues1, nSim)
    return np.mean(abs(actionProbs1 - actionProbs0), axis=0)


def checkminMax(allActions, nSim, tol):
    a = np.reshape(allActions, (t0, nSim, nActions * 2))
    relDiff = ((np.max(a, axis=0) - np.min(a, axis=0))/np.min(a, axis=0))
    return np.all(relDiff < tol, axis=1)

def checkVariance(allActions, cIter, tol):
    h = (1 / nActions) * np.sum((1 / t0) * np.sum(allActions ** 2, axis=0) - ((1 / t0) * np.sum(allActions, axis=0)) ** 2,
                              axis=2)
    v = np.mean(np.var(allActions, axis = 0), axis = 2)
    return np.all(v < tol, axis=1)


plotExpo = []

for alpha in np.linspace(2e-2, 5e-2, num=1):
    for Gamma in np.linspace(-0.5, 0, num=1):

        nSim = initnSim

        payoffs = generateGames(Gamma, nSim, nActions)

        allActions = []
        allVariance = []
        converged = 0

        qValues0 = np.random.rand(2, nActions, nSim)
        qValues1 = np.random.rand(2, nActions, nSim)

        delta_1 = getDelta(qValues0, qValues1, nSim)

        for cIter in tqdm(range(nIter)):

            if cIter == t0:
                allActions = []

            if cIter % t0 == 0 and cIter != 0 and cIter != t0:

                vars = checkminMax(np.array(allActions), nSim, 1e-2)
                idx = np.where(vars)
                qValues0 = np.delete(qValues0, idx, axis=2)
                payoffs = [np.delete(payoffs[0], idx, axis=2), np.delete(payoffs[1], idx, axis=2)]
                nSim -= len(idx[0])
                converged += len(idx[0])
                allActions = []

            if nSim <= 0:
                break

            qValues0 = qUpdate(qValues0, payoffs, nSim)
            allActions += [getActionProbs(qValues0, nSim)]

        plotExpo += [np.array([alpha, Gamma, converged / initnSim])]

plotExpo = np.array(plotExpo)

a = np.flip(plotExpo[:, 2].reshape(10, 10).T, axis=0)
sns.heatmap(a, vmin=0, vmax=1, xticklabels=np.linspace(1e-2, 5e-2, num=10),
            yticklabels=np.linspace(-1, 0, num=10)[-1::-1])