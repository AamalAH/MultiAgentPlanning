import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from os import mkdir

from pydmd import DMD

# alpha = .1
gamma = 0.1
Gamma = 0.1
tau = 2
alpha = 0.1

nActions = 5
t0 = 500

initnSim = 1

delta_0 = 1e-3
nSim = 10
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
    cov[:nElements, nElements:] = np.eye( nElements) * gamma  # <a_ij b_ji> = Gamma
    cov[nElements:, :nElements] = np.eye(nElements) * gamma

    rewardAs, rewardBs = np.eye(nAct), np.eye(nAct)

    for i in range(nSim):
        rewards = np.random.multivariate_normal(
            np.zeros(2 * nElements), cov=cov)

        rewardAs = np.dstack( (rewardAs, rewards[0:nElements].reshape((nAct, nAct))))
        # rewardAs = np.dstack((rewardAs, np.array([[1, 5], [0, 3]])))
        rewardBs = np.dstack((rewardBs, rewards[nElements:].reshape((nAct, nAct)).T))
        # rewardBs = np.dstack((rewardBs, np.array([[1, 0], [5, 3]])))
    return [rewardAs[:, :, 1:], rewardBs[:, :, 1:]]


def getActionProbs(qValues, nSim):
    """

    Returns all probabilities for all actions and both agents in all Simulations

    :args
    qValues: 2 x nActions x nSim numpy array showing the qValues of all agents across all Actions and Simulations

    nSim: number of simulations

    :returns
    actionProbs: nSim x 2 x nActions array of probabilities
    """

    partitionFunction = np.sum(np.exp(tau * qValues), axis=1)
    actionProbs = np.array(
        [np.array([np.exp(tau * qValues[p, :, s]) / partitionFunction[p, s] for p in range(2)]) for s in range(nSim)])

    return actionProbs


def qUpdate(qValues, payoffs):
    """
    Updates the qValues according the the Q-Update equation with the parameters defined in the body of the code.

    :args
    qValues: 2 x nActions x nSim numpy array showing the qValues of all agents across all Actions and Simulations

    payoffs: list of matrices each nAct x nAct x nSim

    :returns

    qValues: 2 x nActions x nSim numpy array showing the qValues of all agents across all Actions and Simulations
    """

    actionProbs = getActionProbs(qValues, nSim)

    boltzmannChoices = np.array(
        [[np.random.choice(list(range(nActions)), p=actionProbs[s, p, :]) for p in range(2)] for s in range(nSim)])

    rewardAs = payoffs[0][boltzmannChoices[:, 0],
                          boltzmannChoices[:, 1], (range(nSim))]
    rewardBs = payoffs[1][boltzmannChoices[:, 0],
                          boltzmannChoices[:, 1], (range(nSim))]

    qValues[[0] * nSim, boltzmannChoices[:, 0], (range(nSim))] += alpha * (
            rewardAs - qValues[[0] * nSim, boltzmannChoices[:, 0], (range(nSim))] + gamma * np.max(
        qValues[[0] * nSim, :, (range(nSim))], axis=1))

    qValues[[1] * nSim, boltzmannChoices[:, 1], list(range(nSim))] += alpha * (
            rewardBs - qValues[[1] * nSim, boltzmannChoices[:, 1], (range(nSim))] + gamma * np.max(
        qValues[[1] * nSim, :, (range(nSim))], axis=1))

    return qValues


def stringActions(actionProbs):
    return actionProbs[0].T.reshape((nActions * 2))
    # return actionProbs.reshape((nSim, 2 * nActions))


if __name__ == "__main__":

    plotFractalDim = []

    for alpha in tqdm(np.linspace(1e-2, 5e-2, num=1)):
        for Gamma in np.linspace(-1, 0.5, num=1):
            payoffs = generateGames(Gamma, nSim, nActions)
            allActions = []

            qValues0 = np.random.rand(2, nActions, nSim)

            for cIter in range(nIter):
                qValues0 = qUpdate(qValues0, payoffs)

                if cIter >= 0:

                    allActions += [stringActions(getActionProbs(qValues0, nSim))]

                    if cIter == 1000:
                        dmd = DMD(svd_rank=-1)

                        dmd.fit(np.array(allActions).T)ed 
                        dmd.plot_eigs()
                        plt.show()

                    if cIter % 12000 == 0 and cIter != 0:

                        dmd = DMD(svd_rank=-1)

                        dmd.fit(np.array(allActions).T)

                        fig = plt.figure()
                        ax1 = fig.add_subplot(121)
                        ax2 = fig.add_subplot(122)
                        allActions = np.array(allActions)
                        ax1.plot(allActions[:, 2], allActions[:, 3], 'r--')
                        ax1.set_xlim([0, 1]), ax1.set_ylim([0, 1])

                        A = dmd.reconstructed_data.real

                        ax2.plot(A[2], A[3], 'b--')
                        ax2.set_xlim([0, 1]), ax2.set_ylim([0, 1])

                        # fig2 = plt.figure()
                        # ax12 = fig2.add_subplot(121)
                        # ax22 = fig2.add_subplot(122)
                        #
                        # ax12.plot(allActions[:, 2], allActions[:, 3], 'r--')
                        # ax12.set_xlim([0, 1]), ax12.set_ylim([0, 1])
                        #
                        # A = dmd.reconstructed_data.real
                        #
                        # ax22.plot(A[2], A[3], 'b--')
                        # ax22.set_xlim([0, 1]), ax22.set_ylim([0, 1])

                        plt.show()

                        # allActions = []
