import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from os import mkdir


def fractalDimension(allActions):
    """
    Returns the frac]tal dimension through length calculation of a given trajectory. This is given as

    \frac{log_{10} n}{log_{10} (L/l) + log_{10} n}
    where n = L/u, L is the total length of the curve, l is the maximum distance of any point in the trajectory from the
    start point and u is the average distance between successive points in the trajectory.

    :arg
    allActions: list of actions in the trajectory. Each action is a 3-d array of shape (nSim, nPlayer, nAction)

    :return
    fractalDim: the fractal dimension as given above as a tuple of length nSim
    """

    allActions = np.array(allActions)

    #     Calculate the length of the curve. The curve is through the entire joint strategy space of all agents.
    #     Therefore, norms are taken over all actions and players

    findDistance = lambda point1, point2: np.linalg.norm(point1 - point2, axis=(1, 2))

    avgCoordVector = (np.mean(allActions, axis=0) - allActions[0]).reshape((nSim, 2 * nActions))
    scalarProjection = lambda point: np.einsum('ij,ij ->i', (point - allActions[0]).reshape((nSim, 2 * nActions)),
                                               avgCoordVector)

    allProjections = np.array([scalarProjection(allActions[i]) for i in range(nIter)])

    # allDistances = [findDistance(allActions[i], allActions[i - 1]) for i in range(1, nIter)]
    allDistances = [abs(allProjections[i] - allProjections[i - 1]) for i in range(1, nIter)]

    curveLength = np.sum(np.array(allDistances), axis=0)

    #     Calculate the maximum distance from the start point from all points in the curve
    # maxDist = np.max(np.array([findDistance(allActions[i], allActions[0]) for i in range(1,
    # nIter)]), axis = 0)

    maxDist = np.max([abs(allProjections[i] - allProjections[0]) for i in range(1, nIter)], axis=0)

    #     Calculate the average distance between successive points
    averageDist = np.mean(allDistances)

    #     Determine the fractal dimension
    fractalDim = (np.log10(curveLength / averageDist)) / (
            np.log10(curveLength / maxDist) + np.log10(curveLength / averageDist))

    return fractalDim

if __name__ == "__main__":

    gamma = 0.1
    Gamma = 0.1
    tau = 1

    nActions = 5
    t0 = 500

    initnSim = 1

    nSim = 10
    nIter = int(1.5e3)

    for alpha in tqdm(np.linspace(1e-2, 5e-2, num=10)):
        for Gamma in np.linspace(-1, 1, num=10):
            payoffs = generateGames(Gamma, nSim, nActions)
            allActions = []

            qValues0 = np.random.rand(2, nActions, nSim)

            for cIter in range(nIter):
                qValues0 = qUpdate(qValues0, payoffs)
                allActions += [getActionProbs(qValues0, nSim)]

            plotFractalDim.append(np.array([alpha, Gamma, np.mean(fractalDimension(allActions))]))