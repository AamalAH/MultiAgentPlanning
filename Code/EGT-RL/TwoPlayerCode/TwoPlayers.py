import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

# alpha = .1
gamma = .1
Gamma = 0.1
tau = 0.05

nActions = 5

delta_0 = 1e-3
nSim = 15
nIter = int(1.5e4)


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
        # rewardAs = np.dstack((rewardAs, np.array([[1, 5], [0, 3]])))
        rewardBs = np.dstack((rewardBs, rewards[nElements:].reshape((nAct, nAct)).T))
        # rewardBs = np.dstack((rewardBs, np.array([[1, 0], [5, 3]])))
    return [rewardAs[:, :, 1:], rewardBs[:, :, 1:]]


def getActionProbs(qValues, nSim):
    partitionFunction = np.sum(np.exp(tau * qValues), axis=1)
    actionProbs = np.array(
        [np.array([np.exp(tau * qValues[p, :, s]) / partitionFunction[p, s] for p in range(2)]) for s in range(nSim)])

    return actionProbs


def qUpdate(qValues, payoffs):
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


plotExpo = []

for alpha in np.linspace(1e-2, 5e-2, num=1):
    for Gamma in np.linspace(-0.5, 0, num=1):

        payoffs = generateGames(Gamma, nSim, nActions)
        allActions = []

        qValues0 = np.random.rand(2, nActions, nSim)


        for cIter in tqdm(range(nIter)):
            qValues0 = qUpdate(qValues0, payoffs)
            allActions += [getActionProbs(qValues0, nSim)]

allActions = np.array(allActions)
fig = plt.figure()
ax1 = fig.add_subplot(111)

plt.plot(allActions[:, 0, 0, 0], allActions[:, 0, 1, 0])
plt.rc('axes', labelsize=12)
plt.xlabel('Player 1 Action 1')
plt.ylabel('Player 2 Action 1')
plt.xlim([0.19, 0.21]), plt.ylim([0.19, 0.21]), plt.show()