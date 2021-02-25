import numpy as np
import nashpy as nash
from tqdm import tqdm
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

def initialiseRandomVectors(dim, nInit, nSim):
    pX = np.dstack([np.random.dirichlet(np.ones(dim), size=(nInit)).T for s in range(nSim)])
    qY = np.dstack([np.random.dirichlet(np.ones(dim), size=(nInit)).T for s in range(nSim)])

    return pX, qY

def simulation(gamma, dim, nSim, nInit, nIter):

    A, B = generateGallaGames(gamma, nSim, dim)

    pX, qY = initialiseRandomVectors(dim, nInit, nSim)

    allPX, allQY = np.zeros((dim, nInit, nSim, int(nIter))), np.zeros((dim, nInit, nSim, int(nIter)))

    windowLength = 50

    convergenceWindow = np.zeros((dim, nInit, nSim, windowLength))

    for n in range(1, int(nIter) + 1):
        eX, eY = np.zeros((dim, nInit, nSim)), np.zeros((dim, nInit, nSim))
        brX, brY = np.argmax(np.einsum('ijl,jkl->ikl', A, qY), axis=0), np.argmax(np.einsum('jil,jkl->ikl', B, pX), axis=0)

        for s in range(nSim):
            eX[brX[:, s], range(nInit), s] = 1
            eY[brY[:, s], range(nInit), s] = 1

        pX = (n * pX + eX) / (n + 1)
        qY = (n * qY + eY) / (n + 1)

        allPX[:, :, :, n-1] = pX
        allQY[:, :, :, n-1] = qY

    convergenceWindow = allPX[:, :, :, n-1-windowLength:n-1]

    tol = 5e-2
    notConverged = np.where(np.linalg.norm((convergenceWindow[:, :, :, 0] - convergenceWindow[:, :, :, -1])/convergenceWindow[:, :, :, 0], axis=0) > tol)

    return A, allPX[:, notConverged[0], notConverged[1], :]

def plotOnSimplex(gamma, trajX):
    f, (ax) = plt.subplots(1, 1)

    proj = np.array(
        [[-1 * np.cos(30 / 360 * 2 * np.pi), np.cos(30 / 360 * 2 * np.pi), 0],
         [-1 * np.sin(30 / 360 * 2 * np.pi), -1 * np.sin(30 / 360 * 2 * np.pi), 1]
         ])

    ts = np.linspace(0, 1, 10000)

    e1 = proj @ np.array([ts, 1 - ts, 0 * ts])
    e2 = proj @ np.array([0 * ts, ts, 1 - ts])
    e3 = proj @ np.array([ts, 0 * ts, 1 - ts])

    ax.plot(e1[0], e1[1])
    ax.plot(e2[0], e2[1])
    ax.plot(e3[0], e3[1])

    for i in range(trajX.shape[1]):
        d = proj @ trajX[:, i, :]
        ax.plot(d[0], d[1])
        ax.scatter(d[0, -1], d[1, -1], color='r', marker='+')

    plt.title(str(np.round(gamma, 3)))

    plt.show()

def findNE(A, B):
    rps = nash.Game(A, B)
    eqs = rps.support_enumeration()
    return list(eqs)

if __name__ == '__main__':

    numTests = 10
    dim = 3
    nSim = 100
    nInit = 10
    nIter = 1e4

    for cTest, gamma in enumerate(np.linspace(-1, 1, num=numTests)):

        A, allPX = simulation(gamma, dim, nSim, nInit, nIter)
        if allPX.shape[1] > 0:
            plotOnSimplex(gamma, allPX)