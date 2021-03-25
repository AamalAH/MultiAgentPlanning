import numpy as np
import nashpy as nash
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


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

def findNE(A, B):
    rps = nash.Game(A, B)
    eqs = rps.support_enumeration()
    return list(eqs)

def sampleFromSimplex(dim, nSamples=1.5e3):
    nSamples = int(nSamples)
    return np.random.dirichlet(np.ones(dim), size=nSamples)

def ABar(samples, A, q):
    return np.max(np.einsum('ni,ijs,jcs->ncs', samples, A, q), axis=0)

def BBar(samples, B, p):
    return np.max(np.einsum('ics,ijs,nj->ncs', p, B, samples), axis=0)

def simulation(gamma, dim, nSim, nInit, nIter):

    samples = sampleFromSimplex(dim, nSamples=1e4)

    A, B = generateGallaGames(gamma, nSim, dim)

    allNE = [findNE(A[:, :, i], B[:, :, i]) for i in range(nSim)]

    simsToKeep = np.where(np.array([len(sim) for sim in allNE]) != 0)[0]
    allNE = np.array(allNE)[simsToKeep].tolist()

    A, B = A[:, :, simsToKeep], B[:, :, simsToKeep]

    nSim = len(simsToKeep)

    pX, qY = initialiseRandomVectors(dim, nInit, nSim)

    allPX, allQY = np.zeros((dim, nInit, nSim, int(nIter))), np.zeros((dim, nInit, nSim, int(nIter)))
    erX, erY = np.einsum('ais,ajs,jis->is', pX, A, qY), np.einsum('ais,ajs,jis->is', pX, B, qY)
    allerX, allerY = np.zeros((nInit, nSim, nIter)), np.zeros((nInit, nSim, nIter))

    windowLength = 100

    convergenceWindow = np.zeros((dim, nInit, nSim, windowLength))

    for n in tqdm(range(1, int(nIter) + 1)):
        eX, eY = np.zeros((dim, nInit, nSim)), np.zeros((dim, nInit, nSim))
        brX, brY = np.argmax(np.einsum('ijl,jkl->ikl', A, qY), axis=0), np.argmax(np.einsum('jil,jkl->ikl', B, pX), axis=0)

        for s in range(nSim):
            eX[brX[:, s], range(nInit), s] = 1
            eY[brY[:, s], range(nInit), s] = 1

        pX = (n * pX + eX) / (n + 1)
        qY = (n * qY + eY) / (n + 1)

        allPX[:, :, :, n-1] = pX
        allQY[:, :, :, n-1] = qY

        erX = (n * erX + np.einsum('ais,ajs,jis->is', eX, A, eY)) / (n + 1)
        erY = (n * erY + np.einsum('ais,ajs,jis->is', eX, B, eY)) / (n + 1)

        allerX[:, :, n - 1] = erX
        allerY[:, :, n - 1] = erY

    convergenceWindowX = allPX[:, :, :, n-1-windowLength:n-1]
    convergenceWindowY = allQY[:, :, :, n-1-windowLength:n-1]

    tol = 5e-2
    notConvergedX = np.where(np.any(((np.max(convergenceWindowX, axis=3) - np.min(convergenceWindowX, axis=3))/np.min(convergenceWindowX, axis=3)) > tol, axis=0))
    notConvergedY = np.where(np.any(((np.max(convergenceWindowY, axis=3) - np.min(convergenceWindowY, axis=3))/np.min(convergenceWindowY, axis=3)) > tol, axis=0))

    notConverged = [np.hstack((notConvergedX[0], notConvergedY[0])), np.hstack((notConvergedX[1], notConvergedY[1]))]

    return A[:, :, notConverged[1]], B[:, :, notConverged[1]], allPX[:, notConverged[0], notConverged[1], :], allQY[:, notConverged[0], notConverged[1], :], allerX[notConverged[0], notConverged[1], :], allerY[notConverged[0], notConverged[1], :], notConverged

def plotOnSimplex(gamma, trajX, trajY):
    f, (ax) = plt.subplots(1, 2)

    proj = np.array(
        [[-1 * np.cos(30 / 360 * 2 * np.pi), np.cos(30 / 360 * 2 * np.pi), 0],
         [-1 * np.sin(30 / 360 * 2 * np.pi), -1 * np.sin(30 / 360 * 2 * np.pi), 1]
         ])

    ts = np.linspace(0, 1, 10000)

    e1 = proj @ np.array([ts, 1 - ts, 0 * ts])
    e2 = proj @ np.array([0 * ts, ts, 1 - ts])
    e3 = proj @ np.array([ts, 0 * ts, 1 - ts])

    ax[0].plot(e1[0], e1[1], 'k-', alpha = 0.3)
    ax[0].plot(e2[0], e2[1], 'k-', alpha = 0.3)
    ax[0].plot(e3[0], e3[1], 'k-', alpha = 0.3)

    ax[1].plot(e1[0], e1[1], 'k-', alpha=0.3)
    ax[1].plot(e2[0], e2[1], 'k-', alpha=0.3)
    ax[1].plot(e3[0], e3[1], 'k-', alpha=0.3)

    for i in range(trajX.shape[1]):
        d = proj @ trajX[:, i, :]
        ax[0].plot(d[0], d[1], '--', alpha=0.6)
        ax[0].scatter(d[0, -1], d[1, -1], marker='+')

        d = proj @ trajY[:, i, :]
        ax[1].plot(d[0], d[1], '--', alpha=0.6)
        ax[1].scatter(d[0, -1], d[1, -1], marker='+')

    plt.title(str(np.round(gamma, 3)))

    plt.show()

    return (f, (ax))

def getAllPayoffs(dim, erX, erY, allPX, allQY, case, A, B):
    allNE = findNE(A[:, :, case], B[:, :, case])
    allNEPayoffs = [(allNE[i][0] @ A[:, :, case] @ allNE[i][1], allNE[i][0] @ B[:, :, case] @ allNE[i][1]) for i in
                range(len(allNE))]

    samples = sampleFromSimplex(dim, nSamples=1e4)
    aBar = np.max(samples @ A[:, :, case] @ allQY[:, case, -1])
    bBar = np.max(allPX[:, case, -1].T @ B[:, :, case] @ samples.T)

    return allNEPayoffs, (erX[case, -1], erY[case, -1]), (aBar, bBar)

def plotPayoffs(erX, erY, A, B, case, nIter):
    f, (ax) = plt.subplots(1, 2)

    allNE = findNE(A[:, :, case], B[:, :, case])
    allNEPayoffs = [(allNE[i][0] @ A[:, :, case] @ allNE[i][1], allNE[i][0] @ B[:, :, case] @ allNE[i][1]) for i in range(len(allNE))]

    ax[0].plot(range(nIter), erX[case, :], '--')
    ax[1].plot(range(nIter), erY[case, :], '--')
    for i in range(len(allNEPayoffs)):
        ax[0].plot(range(nIter), [allNEPayoffs[i][0]]*nIter, 'k-')
        ax[1].plot(range(nIter), [allNEPayoffs[i][1]]*nIter, 'k-')

    plt.title(str(np.round(gamma, 3)))

    plt.show()

    return (f, (ax))

def findNE(A, B):
    rps = nash.Game(A, B)
    eqs = rps.support_enumeration()
    return list(eqs)

def writeData(dataNo, gamma, dim, erX, erY, allPX, allQY, case, A, B, nIter):
    plotPayoffs(erX, erY, A, B, case, nIter)

    allNEPayoffs, rewards, bars = getAllPayoffs(dim, allerX, allerY, allPX, allQY, case, A, B)

    data = {'data': dataNo,
            'gamma': np.round(gamma, 3),
            'allNEPayoffsX': [allNEPayoffs[i][0] for i in range(len(allNEPayoffs))],
            'rewardX': rewards[0],
            'ABar': bars[0],
            'allNEPayoffsY': [allNEPayoffs[i][1] for i in range(len(allNEPayoffs))],
            'rewardY': rewards[1],
            'BBar': bars[1],
            'A': A[:, :, case],
            'B': B[:, :, case]
            }
    with open('ExperimentData/Data.txt', 'a') as file:
        file.write(str(data) + '\n')

if __name__ == '__main__':

    numTests = 20
    dim = 3
    nSim = 150
    nInit = 50
    nIter = 5e4
    nIter = int(nIter)

    gammas = np.linspace(-1, 1, num=numTests)

    gamma = gammas[1]

    A, B, allPX, allQY, allerX, allerY, notConverged = simulation(gamma, dim, nSim, nInit, nIter)
    if allPX.shape[1] > 0:
        plotOnSimplex(gamma, allPX, allQY)
        pass