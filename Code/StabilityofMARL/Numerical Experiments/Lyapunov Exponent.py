import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from os import mkdir
from qUpdater.TwoPlayers import *

gamma = 0.1
Gamma = 0.1
tau = 1

nActions = 5
t0 = 500

initnSim = 1

nSim = 10
nIter = int(1.5e3)


def KantzLCE(data):


for alpha in tqdm(np.linspace(1e-2, 5e-2, num=1)):
    for Gamma in np.linspace(-1, 1, num=1):
        payoffs = generateGames(Gamma, nSim, nActions)
        allActions = []

        qValues0 = np.random.rand(2, nActions, nSim)

        for cIter in range(nIter):
            qValues0 = qUpdate(qValues0, payoffs, nSim, nActions, agentParams=(alpha, tau, gamma))
            allActions += [getActionProbs(qValues0, nSim, agentParams=(alpha, tau, gamma))]