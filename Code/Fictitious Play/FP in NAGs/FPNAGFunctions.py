import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy.sparse as sps

def generateGallaGames(gamma, nSim, dim):
    nElements = dim ** 2  # number of payoff elements in the matrix

    cov = np.eye(2 * nElements)  # <a_ij^2> = <b_ji^2> = 1
    cov[:nElements, nElements:] = np.eye(nElements) * gamma  # <a_ij b_ji> = Gamma
    cov[nElements:, :nElements] = np.eye(nElements) * gamma

    rewardAs, rewardBs = np.eye(dim), np.eye(dim)

    for i in range(nSim):
        rewards = np.random.multivariate_normal(np.zeros(2 * nElements), cov=cov)

        rewardAs = np.dstack((rewardAs, rewards[0:nElements].reshape((dim, dim))))
        rewardBs = np.dstack((rewardBs, rewards[nElements:].reshape((dim, dim))))

    return [rewardAs[:, :, 1:], rewardBs[:, :, 1:]]

# def generateGames(gamma, dim, nAgents):
#     """
#     Draw a random payoff matrix from a multivariate Gaussian, currently the unscaled version.
#     gamma: Choice of co-operation parameter
#     nAct: Number of actions in the game
#     [reward1s, reward2s]: list of payoff matrices
#     """
#
#     nElements = dim ** nAgents  # number of payoff elements in a player's matrix
#
#     cov = sps.lil_matrix(sps.eye(nAgents * nElements))
#     idX = list(itertools.product(list(range(1, dim + 1)), repeat=nAgents)) * nAgents
#     for idx in range(nElements):
#         for b in range(1, nAgents):
#             rollPlayers = np.where([np.all(idX[i] == np.roll(idX[idx], -1 * b)) for i in range(len(idX))])[0][b]
#             cov[idx, rollPlayers], cov[rollPlayers, idx] = gamma/(nAgents - 1), gamma/(nAgents - 1)
#
#     cov = cov.toarray()
#
#     rewards = np.random.multivariate_normal(np.zeros(nAgents * nElements), cov=cov)
#     each = np.array_split(rewards, nAgents)
#
#     return np.array(each)

def generateZeroSumNetworkGame(dim, nAgents):

    G = np.random.randint(0, 10, size=(dim, dim, nAgents))
    G = np.dstack([G[:, :, i]/np.sum(G, axis=2) for i in range(nAgents)])
    G = np.dstack([G[:, :, i] - (1/nAgents * np.ones((dim, dim))) for i in range(nAgents)])

    return G

def initialiseRandomVectors(dim, nInit, nAgents):

    # strategies = np.zeros((dim, nInit * nAgents))
    # strategies[np.random.randint(0, dim, size=(nInit*nAgents)), range(nInit*nAgents)] = 1
    #
    # return np.reshape(strategies, (dim, nAgents, nInit))
    return np.random.dirichlet(np.ones(dim), size=(nAgents, nInit)).transpose(2, 0, 1)

def aggregateStrategies(P, dim, nAgents, nInit, strats):
    z = np.kron(P, np.eye(dim)) @ np.reshape(strats.transpose(1, 0, 2), (nAgents * dim, nInit))

    return z.reshape((3, nAgents, nInit), order='F')

def simulate(dim, nAgents, nInit = 50, nIter = 1e4):

    nIter = int(nIter)
        
    P = (np.eye(nAgents) == False) * (1/(nAgents - 1))

    G = generateZeroSumNetworkGame(dim, nAgents)
    strats = initialiseRandomVectors(dim, nInit, nAgents)
    allStrat = np.zeros((dim, nAgents, nInit, nIter))

    allref = np.zeros((dim, nAgents, nInit, nIter))

    for n in range(1, nIter + 1):
        ref = aggregateStrategies(P, dim, nAgents, nInit, strats)
        e = np.zeros((dim, nInit * nAgents))
        BR = np.argmax(np.einsum('ijs,jns->ins', G, ref.transpose(0, 2, 1)), axis=0).reshape((nInit * nAgents))

        e[BR, range(nInit * nAgents)] = 1
        e = e.reshape((dim, nAgents, nInit), order='F')

        strats = (n*strats + e) / (n + 1)

        allref[:, :, :, n-1] = ref
        allStrat[:, :, :, n-1] = strats

    return allStrat

def plotOnSimplex(trajX, nAgents, nInit):
    f, ax = plt.subplots(1, nAgents, sharex='all', sharey='all')

    proj = np.array(
        [[-1 * np.cos(30 / 360 * 2 * np.pi), np.cos(30 / 360 * 2 * np.pi), 0],
         [-1 * np.sin(30 / 360 * 2 * np.pi), -1 * np.sin(30 / 360 * 2 * np.pi), 1]
         ])

    ts = np.linspace(0, 1, 10000)

    e1 = proj @ np.array([ts, 1 - ts, 0 * ts])
    e2 = proj @ np.array([0 * ts, ts, 1 - ts])
    e3 = proj @ np.array([ts, 0 * ts, 1 - ts])

    for cAgent in range(nAgents):

        ax[cAgent].plot(e1[0], e1[1], 'k')
        ax[cAgent].plot(e2[0], e2[1], 'k')
        ax[cAgent].plot(e3[0], e3[1], 'k')

        for i in range(nInit):
            d = proj @ trajX[:, cAgent+15, i, :]
            ax[cAgent].plot(d[0], d[1], '--', alpha=0.6)
            ax[cAgent].scatter(d[0, -1], d[1, -1], color='r', marker='+')

    plt.show()

if __name__ == '__main__':
    dim = 3
    nInit = 50
    nAgents = 25

    plotOnSimplex(simulate(dim, nAgents), 2, nInit)
