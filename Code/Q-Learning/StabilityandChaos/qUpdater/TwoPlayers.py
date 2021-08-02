import numpy as np

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


def getActionProbs(qValues, nSim, agentParams):
    tau = agentParams[1]

    partitionFunction = np.sum(np.exp(tau * qValues), axis=1)
    actionProbs = np.array(
        [np.array([np.exp(tau * qValues[p, :, s]) / partitionFunction[p, s] for p in range(2)]) for s in range(nSim)])

    return actionProbs


def qUpdate(qValues, payoffs, nSim, nActions, agentParams):
    (alpha, tau, gamma) = agentParams

    actionProbs = getActionProbs(qValues, nSim, agentParams)

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

