#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 20:49:19 2020

@author: aamalh
"""

import numpy as np
import itertools
from time import time

nAct = 2
nPlayers = 2
nSim = 1

alpha = 0.5
gamma = 0.5
tau = 1
Gamma = -1


def generateGames(gamma, nSim, nAct):
    """
    Draw a random payoff matrix from a multivariate Gaussian, currently the unscaled version.

    gamma: Choice of co-operation parameter
    nAct: Number of actions in the game

    [reward1s, reward2s]: list of payoff matrices
    """

    nElements = nAct ** nPlayers  # number of payoff elements in the matrix

    cov = np.eye(2 * nElements)  # <a_ij^2> = <b_ji^2> = 1
    cov[:nElements, nElements:] = np.eye(nElements) * gamma  # <a_ij b_ji> = Gamma
    cov[nElements:, :nElements] = np.eye(nElements) * gamma

    for i in range(nSim):

        rewards = np.random.multivariate_normal(np.zeros(2 * nElements), cov=cov)

    return np.array([1, 5, 0, 3, 1, 5, 0, 3])

def getActionProbs(qValues):
    partitionFunction = np.sum(np.exp(tau * qValues0.reshape((2, 2))), axis = 1)
    actionProbs = [np.exp(tau * qValues0[i:i + nAct])/partitionFunction[int(i/2)] for i in range(0, nPlayers + 1, nAct)]

    return np.array(actionProbs).squeeze()

rewards = generateGames(Gamma, nSim, nAct)
qValues0 = np.random.rand(nPlayers * nAct, nSim)

iterTimes = []
allActions = []

for cIter in range(1000):

    start = time()

    actionProbs = getActionProbs(qValues0)
    bChoices = [np.random.choice([0, 1], p = actionProbs[p]) for p in range(nPlayers)]
    
    rows = [i + j + 1 for (j, i) in enumerate(bChoices)]
    rows[0] -= 1
    idX = list(itertools.product(list(range(0, nAct)), repeat = nPlayers))
    idX = [list(i) for i in idX]
    
    cols = [np.where([i == np.roll(bChoices, int(-1 * p)).tolist() for i in idX])[0][0] + 2 for p in range(2)]
    cols[0] -= nAct**nPlayers
    
    Mat = np.zeros((2 * 2, 2 * 2 ** 2))
    Mat[rows, cols] = 1
    
    Mat2 = np.zeros((4, 4))
    Mat2[rows, rows] = 1
    
    Mat3 = np.zeros((4, 4))
    maxs = np.argmax(actionProbs, axis = 1) + np.array([2 * p for p in range(2)])
    Mat3[rows, maxs] = 1
    
    qValues0 += alpha * (np.expand_dims(Mat @ rewards, 1) - (Mat2 @ qValues0) + gamma * (Mat2 @ qValues0))

    allActions += [getActionProbs(qValues0)]

    iterTimes.append(time() - start)



print(np.mean(iterTimes)) 