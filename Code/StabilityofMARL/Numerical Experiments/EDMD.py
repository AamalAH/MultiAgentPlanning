import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
from itertools import product
from qUpdater import TwoPlayers
"""
EDMD Algorithm: 

1. Define a series of observables: let's say Hermite Polynomials up to order 5 + [x, v] + [1]
2. Collect a set of data pairs X_n-1 and X_n
3. Apply the observable functions on the datasets to get \Phi(X), \Phi(Y)
4. Compute G and A from EDMD paper
5. Compute K

Predicting the flow field:

Apply K on the observable for some initial condition. 
Extract the flow field using (3.2) and (3.3) of the KMD for Algs paper
"""

def eval_Phi(x, maxOrder = 5):
    lenData = len(x)
    allHermiteOrders = list(product(range(maxOrder), repeat=lenData))
    allHermites = np.array([sps.eval_hermite(o,x) for o in range(maxOrder)])

    allCombos = [np.prod(allHermites[oList, range(lenData)]) for oList in allHermiteOrders]

    return allCombos

def generatePhi(X, maxOrder=5, isX = True):
    if isX:
        return np.array(
            [eval_Phi(X[i], maxOrder=maxOrder) + [*X[i]] + [*(X[i] - X[i-1]/1e-2)] + [i] for i in range(1, len(X))])
    else:
        return np.array(
            [eval_Phi(X[i], maxOrder=maxOrder) + [*X[i]] + [*(X[i] - X[i - 1] / 1e-2)] + [i+1] for i in range(1, len(X))])

def generateX_Y(gameParams, agentParams, sizeX):
    (Gamma, nSim, nActions) = gameParams
    (alpha, tau, gamma) = agentParams

    payoffs = TwoPlayers.generateGames(*gameParams)


    qValues = np.random.rand(2, nActions, nSim)

    allActions = [TwoPlayers.getActionProbs(qValues, nSim, agentParams).reshape((2 ** nActions))]

    for cIter in range(sizeX + 1):
        qValues = TwoPlayers.qUpdate(qValues, payoffs, nSim, nActions, agentParams)
        allActions += [TwoPlayers.getActionProbs(qValues, nSim, agentParams).reshape((2 ** nActions))]

    return np.array(allActions[:-1]).squeeze(), np.array(allActions[1:]).squeeze()


if __name__ == "__main__":

    Gamma, nSim, nActions = -0.5, 1, 2
    alpha, tau, gamma = 1e-2, 5e-2, 0.1

    nX = 100

    X, Y = generateX_Y((Gamma, nSim, nActions), (alpha, tau, gamma), nX)
    PX, PY = generatePhi(X), generatePhi(Y, isX=False)

    G = ( 1 / nX ) * PX.T @ PX
    A = ( 1 / nX ) * PX.T @ PY

    K = np.linalg.pinv(G) @ A