#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 22:24:55 2020

@author: aamalh
"""

import numpy as np
import itertools
import scipy.sparse as sps
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

t0 = 500
alpha = 4e-2
gamma = .1
Gamma = -1
tau = 5e-2

delta_0 = 1e-3
initnSim = 10
nIter = int(1e3)
nTests = 10

nPlayers = 3
nActions = 5
nElements = nActions ** nPlayers

def generateGames(gamma, nSim, nAct, nPlayers):
    """
    Draw a random payoff matrix from a multivariate Gaussian, currently the unscaled version.
    gamma: Choice of co-operation parameter
    nAct: Number of actions in the game
    [reward1s, reward2s]: list of payoff matrices
    """

    nElements = nAct ** nPlayers  # number of payoff elements in a player's matrix

    cov = sps.lil_matrix(sps.eye(nPlayers * nElements))
    idX = list(itertools.product(list(range(1, nAct + 1)), repeat=nPlayers)) * nPlayers
    for idx in range(nElements):
        for b in range(1, nPlayers):
            rollPlayers = np.where([np.all(idX[i] == np.roll(idX[idx], -1 * b)) for i in range(len(idX))])[0][b]
            cov[idx, rollPlayers], cov[rollPlayers, idx] = gamma/(nPlayers - 1), gamma/(nPlayers - 1)

    cov = cov.toarray()

    rewards = np.zeros((nPlayers * nElements))

    for i in range(nSim):
        rewards = np.hstack((rewards, np.random.multivariate_normal(np.zeros(nPlayers * nElements), cov=cov)))

    return rewards[nPlayers * nElements: ]

def getActionProbs(qValues):
   
    partitionFunction = np.sum(np.exp(tau * qValues0.reshape((nPlayers * nTests * nSim, nActions))), axis = 1)
    actionProbs = [np.exp(tau * qValues0[i:i + nActions])/partitionFunction[int(i/nActions)] for i in range(0, len(qValues0), nActions)]

    return np.array(actionProbs).squeeze()

def checkConvergence(x, xOld, tol):
  """
    Evaluates the distance between action probabilities and checks if the change is
    below some tolerance.

    params
    actionProb: Agent action probabilities at current time step
    oldxBar: Previous action probabilities
    tol: tolerance level

    returns

    (normStep1 < tol) and (normStep2 < tol): Whether change in both agents' action 
    probabilities is below a tolerance.
  """
  
  normSteps = np.linalg.norm(actionProbs - xOld, axis = 1)
  return np.all(normSteps < tol)


def qUpdate(qValues, payoffs, nSim):

    actionProbs = getActionProbs(qValues)

    bChoices = [np.random.choice(list(range(nActions)), p=actionProbs[p]) for p in range(nPlayers * nTests * nSim)]
    bC = np.array(bChoices).reshape((nTests * nSim, nPlayers))

    rows = [i + j * nActions for (j, i) in enumerate(bChoices)]

    idX = list(itertools.product(list(range(0, nActions)), repeat=nPlayers))
    idX = [list(i) for i in idX]

    cols = np.array([[np.where([i == np.roll(bC[s], int(-1 * p)).tolist() for i in idX])[0][
                          0] + j * nActions ** nPlayers for (j, p) in enumerate(range(nPlayers))] for s in
                     range(nTests * nSim)])
    cols += np.repeat(
        np.expand_dims(np.array([[j * nPlayers * nActions ** nPlayers] * nTests for j in range(nSim)]).flatten(),
                       axis=1), 3, axis=1)
    cols = cols.reshape((nPlayers * nTests * nSim,))

    maxs = np.argmax(actionProbs, axis=1) + np.array([nActions * p for p in range(nPlayers * nTests * nSim)])

    Mat = sps.coo_matrix(([1] * nPlayers * nTests * nSim, (rows, cols.squeeze())),
                         shape=((nPlayers * nActions * nTests * nSim, nSim * nPlayers * nActions ** nPlayers)))
    Mat2 = sps.coo_matrix(([1] * nPlayers * nTests * nSim, (rows, rows)),
                          shape=((nTests * nPlayers * nActions * nSim, nTests * nPlayers * nActions * nSim)))
    Mat3 = sps.coo_matrix(([1] * nPlayers * nTests * nSim, (rows, maxs)),
                          shape=((nPlayers * nActions * nTests * nSim, nTests * nPlayers * nActions * nSim)))
    Mat, Mat2, Mat3 = sps.lil_matrix(Mat), sps.lil_matrix(Mat2), sps.lil_matrix(Mat3)

    qValues += alpha * ((Mat @ payoffs) - (Mat2 @ qValues) + gamma * (Mat3 @ qValues))

    return qValues

def checkminMax(allActions, nSim, tol):
    a = np.reshape(allActions, (t0, nSim, nActions * 2))
    relDiff = ((np.max(a, axis=0) - np.min(a, axis=0))/np.min(a, axis=0))
    return np.all(relDiff < tol, axis=1)


for alpha in tqdm(np.linspace(1e-2, 5e-2, num=10)):
    for Gamma in np.linspace(-1, 0, num=10):

        allActions = []
        nSim = initnSim
        converged = 0

        payoffs = generateGames(Gamma, nSim, nActions, nPlayers)
        qValues0 = np.random.rand(nTests * nPlayers * nActions * nSim)

        start = time()

        for cIter in range(int(1e3)):

            if cIter == t0:
                allActions = []

            if cIter%t0 == 0 and cIter != 0 and cIter != t0:
                vars = checkminMax(np.array(allActions), nSim, 1e-2)
                idx = np.where(vars)
                qValues0 = [np.delete(qValues0[nTests * nPlayers * nActions * i:nTests * nPlayers * nActions * (i+1)]) for i in idx]
                payoffs = [np.delete(payoffs[nPlayers * nElements * i:nPlayers *nElements* (i+1)]) for i in idx]
                nSim -= len(idx[0])
                converged += len(idx[0])
                allActions = []

            if nSim <= 0:
                break

            allActions += [getActionProbs(qValues0)]
            qValues0 = qUpdate(qValues0, payoffs, nSim)

