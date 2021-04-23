import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy.sparse as sps

# def generateGallaGames(gamma, nSim, dim):
#     nElements = dim ** 2  # number of payoff elements in the matrix
#
#     cov = np.eye(2 * nElements)  # <a_ij^2> = <b_ji^2> = 1
#     cov[:nElements, nElements:] = np.eye(nElements) * gamma  # <a_ij b_ji> = Gamma
#     cov[nElements:, :nElements] = np.eye(nElements) * gamma
#
#     rewardAs, rewardBs = np.eye(dim), np.eye(dim)
#
#     for i in range(nSim):
#         rewards = np.random.multivariate_normal(np.zeros(2 * nElements), cov=cov)
#
#         rewardAs = np.dstack((rewardAs, rewards[0:nElements].reshape((dim, dim))))
#         rewardBs = np.dstack((rewardBs, rewards[nElements:].reshape((dim, dim))))
#
#     return [rewardAs[:, :, 1:], rewardBs[:, :, 1:]]

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

def generateNetworkGame(dim, nAgents):

    G = np.random.randint(0, 10, size=(dim, dim, nAgents))
    G = np.dstack([G[:, :, i]/np.sum(G, axis=2) for i in range(nAgents)])
    G = np.dstack([G[:, :, i] - (1/nAgents * np.ones((dim, dim))) for i in range(nAgents)])

    return G

def generateThreePlayerChain(dim, W):
    """
    Generates an NA game formed of three players in a chain
    :param dim: number of strategies
    :param W: aggregation matrix
    :return: dim x dim x 3 matrix, with each layer corresponding to a payoff matrix
    """
    B = np.random.rand(dim, dim)
    A = -W[1, 0] * B.T
    C = -W[1, 2] * B.T
    return np.dstack((A, B, C))

def initialiseRandomVectors(dim, nInit, nAgents):
    """
    generates a set of initial conditions for each agent
    :param dim: number of strategies for each agent
    :param nInit: number of initial conditions
    :param nAgents: number of agents
    :return: dim x nAgents x nInit vector of strategies, with each row corresponding to a pure strategy vector
    """
    return np.random.dirichlet(np.ones(dim), size=(nAgents, nInit)).transpose(2, 0, 1)
    # allStrat = np.zeros((dim, nAgents * nInit))
    # init = np.random.randint(0, dim, size=(nAgents * nInit))
    # allStrat[init, range(nAgents * nInit)] = 1
    # return allStrat.reshape((dim, nAgents, nInit))

def initialiseAggregationMatrix(nAgents):
    """
    Generates a random aggregation matrix which satisfies w_ii = 0 for all i
    :param nAgents: number of agents in the NA Game
    :return: nAgents x nAgents aggregation matrix which is row-stochastic
    """

    W = np.random.rand(nAgents, nAgents)
    W -= np.diag(W) * np.eye(nAgents)
    W /= np.sum(W, axis=0)
    return W.T

def aggregateStrategies(P, dim, nAgents, nInit, strats):
    z = np.kron(P, np.eye(dim)) @ np.reshape(strats.transpose(1, 0, 2), (nAgents * dim, nInit))

    return z.reshape((3, nAgents, nInit), order='F')

def addnoise(ref, sigma):
    return ref + sigma * np.random.standard_normal(size = ref.shape)

def simulate(dim, game, W, nAgents, nInit = 50, nIter = 1e4, noise=False, sigma = 1):

    nIter = int(nIter)

    G = game
    strats = initialiseRandomVectors(dim, nInit, nAgents)
    allStrat = np.zeros((dim, nAgents, nInit, nIter))

    allref = np.zeros((dim, nAgents, nInit, nIter))

    for n in range(1, nIter + 1):
        ref = aggregateStrategies(W, dim, nAgents, nInit, strats)
        if noise:
            ref = addnoise(ref, sigma)
        e = np.zeros((dim, nInit * nAgents))
        BR = np.argmax(np.einsum('ijs,jns->ins', G, ref.transpose(0, 2, 1)), axis=0).reshape((nInit * nAgents))

        e[BR, range(nInit * nAgents)] = 1
        e = e.reshape((dim, nAgents, nInit), order='F')

        strats = (n*strats + e) / (n + 1)

        allref[:, :, :, n-1] = ref
        allStrat[:, :, :, n-1] = strats

    return allStrat



if __name__ == '__main__':
    dim = 3
    nInit = 50
    nAgents = 10
