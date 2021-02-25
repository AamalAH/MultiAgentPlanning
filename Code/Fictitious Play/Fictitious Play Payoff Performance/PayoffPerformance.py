# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from tqdm import tqdm
import nashpy as nash
import seaborn as sns
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


def findNE(A, B):
    rps = nash.Game(A, B)
    eqs = rps.support_enumeration()
    return list(eqs)


def initialiseRandomVectors(dim, nInit, nSim):
    pX = np.dstack([np.random.dirichlet(np.ones(dim), size=(nInit)).T for s in range(nSim)])
    qY = np.dstack([np.random.dirichlet(np.ones(dim), size=(nInit)).T for s in range(nSim)])

    return pX, qY


def findclosestNEPayoff(pX, qY, allNE, A, B, nInit, nSim):
    closestNE = np.vstack([np.argmin(
        np.vstack([np.linalg.norm(pX[:, :, s].T - allNE[s][i][0], axis=1) for i in range(len(allNE[s]))]), axis=0) for s
                           in range(nSim)]).T
    # closestNE = np.argmin(distancetoNE, axis=0)

    NEX = np.vstack(
        [[np.dot(allNE[s][closestNE[c, s]][0], A[:, :, s] @ allNE[s][closestNE[c, s]][1]) for c in range(nInit)] for s
         in range(nSim)]).T

    closestNE = np.vstack([np.argmin(
        np.vstack([np.linalg.norm(qY[:, :, s].T - allNE[s][i][1], axis=1) for i in range(len(allNE[s]))]), axis=0) for s
        in range(nSim)]).T

    NEY = np.vstack(
        [[np.dot(allNE[s][closestNE[c, s]][0], B[:, :, s] @ allNE[s][closestNE[c, s]][1]) for c in range(nInit)] for s
         in range(nSim)]).T

    return NEX, NEY

def getPerformance(erX, erY, allNEPayoffs, nSim):

    NEX, NEY = getAllNEPayoffs(allNEPayoffs, nSim)

    perfX = np.dstack([erX - neX for neX in NEX])
    perfY = np.dstack([erY - neY for neY in NEY])

    return perfX, perfY

def getAllNEPayoffs(allNEPayoffs, nSim):

    NEX, NEY = [[NEPayoff[0] for NEPayoff in sim] for sim in allNEPayoffs], [[NEPayoff[1] for NEPayoff in sim] for sim in allNEPayoffs]

    mostNE = max([len(n) for n in NEX])

    for cSim in range(nSim):
        NEX[cSim] += [np.inf]*(mostNE - len(NEX[cSim]))
        NEY[cSim] += [np.inf]*(mostNE - len(NEY[cSim]))

    return np.array(NEX).T, np.array(NEY).T

def simulation(gamma, dim, nSim, nInit, nIter):
    A, B = generateGallaGames(gamma, nSim, dim)

    allNE = [findNE(A[:, :, i], B[:, :, i]) for i in range(nSim)]

    simsToKeep = np.where(np.array([len(sim) for sim in allNE]) != 0)[0]
    allNE = np.array(allNE)[simsToKeep].tolist()

    A, B = A[:, :, simsToKeep], B[:, :, simsToKeep]

    allNEPayoffs = [[(np.dot(sim[i][0], A[:, :, s] @ sim[i][1]), np.dot(sim[i][0], B[:, :, s] @ sim[i][1])) for i in
                     range(len(sim))] for s, sim in enumerate(allNE)]

    # NEX = [max([NEPayoff[0] for NEPayoff in sim]) for sim in allNEPayoffs]
    # NEY = [max([NEPayoff[1] for NEPayoff in sim]) for sim in allNEPayoffs]

    nSim = len(simsToKeep)

    """ CODE FOR OSTROVSKI TRIALS

    A, B = np.zeros((3, 3)), np.zeros((3, 3))

    for s in range(nSim):

        A = np.dstack((A, np.array([[-1.353259, -1.268538, 2.572738],
                  [0.162237, -1.800824, 1.584291],
                  [-0.499026, -1.544578, 1.992332]])))

        B = np.dstack((B, np.array([[-1.839111, -2.876997, -3.366031],
                  [-4.801713, -3.854987, -3.758662],
                  [6.740060, 6.590451, 6.898102]])))

    A, B = A[:, :, 1:], B[:, :, 1:]

    EX, EY = np.array([0.288, 0.370, 0.342]), np.array([0.335, 0.327, 0.338])

    NEX, NEY = np.dot(EX, A[:, :, 0] @ EY), np.dot(EX, B[:, :, 0] @ EY)

    """

    pX, qY = np.zeros((dim, nInit, nSim)), np.zeros((dim, nInit, nSim))
    pX[np.random.randint(0, dim, size=(nInit)), range(nInit), :] = 1
    qY[np.random.randint(0, dim, size=(nInit)), range(nInit), :] = 1

    # pX, qY = initialiseRandomVectors(dim, nInit, nSim)

    rX, rY = np.einsum('ais,ajs,jis->is', pX, A, qY), np.einsum('ais,ajs,jis->is', pX, B, qY)
    erX, erY = np.einsum('ais,ajs,jis->is', pX, A, qY), np.einsum('ais,ajs,jis->is', pX, B, qY)

    # allPX, allQY = np.zeros((dim, nInit, nSim, int(nIter))), np.zeros((dim, nInit, nSim, int(nIter)))

    m = 1

    for n in range(1, int(nIter) + 1):
        eX, eY = np.zeros((dim, nInit, nSim)), np.zeros((dim, nInit, nSim))
        brX, brY = np.argmax(np.einsum('ijl,jkl->ikl', A, qY), axis=0), np.argmax(np.einsum('jil,jkl->ikl', B, pX),
                                                                                  axis=0)
        for s in range(nSim):
            eX[brX[:, s], range(nInit), s] = 1
            eY[brY[:, s], range(nInit), s] = 1

        pX = (n * pX + eX) / (n + 1)
        qY = (n * qY + eY) / (n + 1)

        if n > int(2 * nIter / 3):
            rX = (m * rX + np.einsum('ais,ajs,jis->is', pX, A, qY)) / (m + 1)
            rY = (m * rY + np.einsum('ais,ajs,jis->is', pX, B, qY)) / (m + 1)

            erX = (m * erX + np.einsum('ais,ajs,jis->is', eX, A, eY)) / (m + 1)
            erY = (m * erY + np.einsum('ais,ajs,jis->is', eX, B, eY)) / (m + 1)

            m += 1

        # allPX[:, :, :, n-1] = pX
        # allQY[:, :, :, n-1] = qY

    perfX, perfY = getPerformance(erX, erY, allNEPayoffs, nSim)

    return perfX, perfY


def plotInterestingScenario(trajX, trajY):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    proj = np.array(
        [[-1 * np.cos(30 / 360 * 2 * np.pi), np.cos(30 / 360 * 2 * np.pi), 0],
         [-1 * np.sin(30 / 360 * 2 * np.pi), -1 * np.sin(30 / 360 * 2 * np.pi), 1]
         ])

    ts = np.linspace(0, 1, 10000)

    e1 = proj @ np.array([ts, 1 - ts, 0 * ts])
    e2 = proj @ np.array([0 * ts, ts, 1 - ts])
    e3 = proj @ np.array([ts, 0 * ts, 1 - ts])

    ax1.plot(e1[0], e1[1])
    ax1.plot(e2[0], e2[1])
    ax1.plot(e3[0], e3[1])

    ax2.plot(e1[0], e1[1])
    ax2.plot(e2[0], e2[1])
    ax2.plot(e3[0], e3[1])

    d = proj @ trajX
    ax1.plot(d[0], d[1])
    ax1.scatter(d[0, -1], d[1, -1], marker='+')

    d = proj @ trajY
    ax2.plot(d[0], d[1])
    ax2.scatter(d[0, -1], d[1, -1], marker='+')

    plt.show()


def plotOnSimplex(trajX, trajY, nInit=1):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    proj = np.array(
        [[-1 * np.cos(30 / 360 * 2 * np.pi), np.cos(30 / 360 * 2 * np.pi), 0],
         [-1 * np.sin(30 / 360 * 2 * np.pi), -1 * np.sin(30 / 360 * 2 * np.pi), 1]
         ])

    ts = np.linspace(0, 1, 10000)

    e1 = proj @ np.array([ts, 1 - ts, 0 * ts])
    e2 = proj @ np.array([0 * ts, ts, 1 - ts])
    e3 = proj @ np.array([ts, 0 * ts, 1 - ts])

    ax1.plot(e1[0], e1[1])
    ax1.plot(e2[0], e2[1])
    ax1.plot(e3[0], e3[1])

    ax2.plot(e1[0], e1[1])
    ax2.plot(e2[0], e2[1])
    ax2.plot(e3[0], e3[1])

    for i in range(nInit):
        d = proj @ trajX[:, i, :]
        ax1.plot(d[0], d[1])
        ax1.scatter(d[0, -1], d[1, -1], marker='+')

        d = proj @ trajY[:, i, :]
        ax2.plot(d[0], d[1])
        ax2.scatter(d[0, -1], d[1, -1], marker='+')

    plt.show()


if __name__ == "__main__":

    # performanceTrackerX = np.zeros((20, 3))
    # performanceTrackerY = np.zeros((20, 3))
    #
    # i = -1
    # for Gamma in tqdm(np.linspace(-1, 1, num=20)):
    #     i += 1
    #     for j, dim in enumerate(range(3, 6), start=0):
    #         avgRX, avgRY = simulation(Gamma, dim, nInit=50, nSim=50, nIter=1.5e3)
    #
    #         performanceTrackerX[i, j] = np.mean(avgRX)
    #         performanceTrackerY[i, j] = np.mean(avgRY)
    #
    # performanceTrackerX = np.flip(performanceTrackerX, axis=0)
    # performanceTrackerY = np.flip(performanceTrackerY, axis=0)
    #
    # sns.heatmap(performanceTrackerX)
    # plt.show()


    performanceTrackerX, performanceTrackerY = np.zeros((20)), np.zeros((20))
    nNashX, nNashY = np.zeros((20)), np.zeros((20))

    for i, Gamma in tqdm(enumerate(np.linspace(-1, 1, num=20))):
        perfX, perfY = simulation(Gamma, 3, nInit=20, nSim=75, nIter=5e3)

        nNashX[i] = 1 - len(np.where((perfX < -1e-2) & (perfX > -np.inf))[0])/len(np.where(perfX > -np.inf)[0])
        nNashY[i] = 1 - len(np.where((perfY < -1e-2) & (perfY > -np.inf))[0])/len(np.where(perfY > -np.inf)[0])

        performanceTrackerX[i] = np.mean(perfX)
        performanceTrackerY[i] = np.mean(perfY)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(-1, 1, num=20), nNashX, 'b-')
    ax.plot(np.linspace(-1, 1, num=20), nNashY, 'r-')
    plt.show()

    # plt.plot(np.linspace(-1, 1, num=20), performanceTrackerX)
    # plt.show()