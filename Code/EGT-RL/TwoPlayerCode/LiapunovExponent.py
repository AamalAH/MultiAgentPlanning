import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

alpha = .1
gamma = .1
Gamma = 0.1
tau = 1

delta_0 = 1e-3
nSim = 5
nIter = 1e5

def generateGame(gamma, nAct):
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

    rewards = np.random.multivariate_normal(np.zeros(2 * nElements), cov=cov)

    reward1s = rewards[0:nElements].reshape((nAct, nAct))
    reward2s = rewards[nElements:].reshape((nAct, nAct)).T

    return [reward1s, reward2s]

def getActionProbs(qValues):
    partitionFunction = [np.sum([np.exp(tau * i) for i in qValues[p, :]]) for p in range(2)]
    actionProbs = np.array([[np.exp(tau * i) / partitionFunction[p] for i in qValues[p, :]] for p in range(2)])
    
    return actionProbs

def qUpdate(qValues):
    actionProbs = getActionProbs(qValues)

    boltzmannChoices = [np.random.choice([0, 1], p=actionProbs[p, :]) for p in range(2)]

    rewards = [payoffA[boltzmannChoices[0], boltzmannChoices[1]],
                       payoffB[boltzmannChoices[0], boltzmannChoices[1]]]

    qValues[0, boltzmannChoices[0]] += alpha * (rewards[0] - qValues[0, boltzmannChoices[0]] + gamma * max(qValues[0, :]))
    
    qValues[1, boltzmannChoices[1]] += alpha * (rewards[1] - qValues[1, boltzmannChoices[1]] + gamma * max(qValues[1, :]))

    return qValues

all_actionProbs = np.eye(2)

allExpo = np.zeros((10, 10))

i = 0
for alpha in tqdm(np.linspace(0, 1, num=10)):
    j = 1
    for Gamma in np.linspace(-1, 1, num=10):

        for cSim in range(nSim):

            payoffA, payoffB = generateGame(Gamma, 2)

            qValues0 = np.random.rand(2, 2)
            qValues1 = qValues0 + np.array([np.random.choice([-delta_0, delta_0]) for i in range(4)]).reshape((2, 2))

            for cIter in range(int(nIter)):
                start = time()
                qValues0, qValues1 = qUpdate(qValues0), qUpdate(qValues1)
                print(time() - start)
            actionProbs0, actionProbs1 = getActionProbs(qValues0), getActionProbs(qValues1)

            all_actionProbs = np.dstack((all_actionProbs, abs(actionProbs1 - actionProbs0)))

        actionDiff = np.sum(all_actionProbs[:, :, 1:], axis=2)/nSim
        delta_n = np.reshape(actionDiff, (1, 4))

        liapExp = np.max((1/nIter) * np.log(delta_n/delta_0))

        allExpo[10 - j, i] = liapExp

        j += 1
    i += 1