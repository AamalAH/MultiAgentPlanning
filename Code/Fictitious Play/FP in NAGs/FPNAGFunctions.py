import numpy as np
import matplotlib.pyplot as plt

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


def initialiseRandomVectors(dim, nAgents):

    strategies = np.zeros((dim, nAgents))
    strategies[np.random.randint(0, dim, size=(nAgents)), range(nAgents)] = 1
    
    return strategies

def aggregateStrategies(P, dim, nAgents, strategies):

    z = np.kron(P, np.eye(dim)) @ strategies.T.reshape(dim * nAgents)

    return np.vstack([z[i:i+dim] for i in range(0, len(z), dim)]).T

def simulate(dim, nAgents, nIter = 1e4):

    nIter = int(nIter)
        
    P = (np.eye(nAgents) == False) * (1/(nAgents - 1))

    A, B = generateGallaGames(-1, 1, 3)
    A, B = np.squeeze(A), np.squeeze(B)
    strats = initialiseRandomVectors(dim, nAgents)

    allStrats = np.zeros((dim, nAgents, nIter))

    for n in range(1, nIter + 1):
        ref = aggregateStrategies(P, dim, nAgents, strats)
        e = np.zeros((dim, nAgents))
        BR = [np.argmax(A @ ref[:, s]) for s in range(nAgents)]

        e[BR, range(nAgents)] = 1

        strats = (n*strats + e) / (n + 1)

        allStrats[:, :, n-1] = strats

    return allStrats

def plotOnSimplex(trajX):
    f, (ax) = plt.subplots(1, 1)

    proj = np.array(
        [[-1 * np.cos(30 / 360 * 2 * np.pi), np.cos(30 / 360 * 2 * np.pi), 0],
         [-1 * np.sin(30 / 360 * 2 * np.pi), -1 * np.sin(30 / 360 * 2 * np.pi), 1]
         ])

    ts = np.linspace(0, 1, 10000)

    e1 = proj @ np.array([ts, 1 - ts, 0 * ts])
    e2 = proj @ np.array([0 * ts, ts, 1 - ts])
    e3 = proj @ np.array([ts, 0 * ts, 1 - ts])

    ax.plot(e1[0], e1[1], 'k')
    ax.plot(e2[0], e2[1], 'k')
    ax.plot(e3[0], e3[1], 'k')

    for i in range(trajX.shape[1]):
        d = proj @ trajX[:, i, :]
        ax.plot(d[0], d[1], 'r--')
        ax.scatter(d[0, -1], d[1, -1], color='r', marker='+')

    plt.show()

if __name__ == '__main__':
    dim = 3
    nAgents = 10

    plotOnSimplex(simulate(dim, nAgents))
