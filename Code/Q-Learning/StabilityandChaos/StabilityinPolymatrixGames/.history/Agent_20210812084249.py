import numpy as np
from scipy.linalg import block_diag
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generateGames(gamma, nSim, nAct):
    """
    Draw a random payoff matrix from a multivariate Gaussian, currently the unscaled version.

    gamma: Choice of co-operation parameter
    nAct: Number of actions in the game

    [reward1s, reward2s]: list of payoff matrices
    """

    nElements = nAct ** 2  # number of payoff elements in the matrix

    cov = np.eye(2 * nElements)  # <a_ij^2> = <b_ji^2> = 1
    cov[:nElements, nElements:] = np.eye(nElements) * gamma  # <a_ij b_ji> = Gamma
    cov[nElements:, :nElements] = np.eye(nElements) * gamma

    rewardAs, rewardBs = np.eye(nAct), np.eye(nAct)

    for i in range(nSim):
        rewards = np.random.multivariate_normal(np.zeros(2 * nElements), cov=cov)

        rewardAs = np.dstack((rewardAs, rewards[0:nElements].reshape((nAct, nAct))))
        # rewardAs = np.dstack((rewardAs, np.array([[1, 5], [0, 3]])))
        rewardBs = np.dstack((rewardBs, rewards[nElements:].reshape((nAct, nAct)).T))
        # rewardBs = np.dstack((rewardBs, np.array([[1, 0], [5, 3]])))
    return [rewardAs[:, :, 1:], rewardBs[:, :, 1:]]

def checkVar(checkWindow, windowSize, tol):
    C = checkWindow.reshape((nAct * nAgents, nInit, windowSize))
    return np.mean(np.var(C, axis=2), axis=0) < tol

def checkMinMax(checkWindow, windowSize, tol):
    C = np.transpose(checkWindow, (0, 3, 1, 2, 4))
    C = C.reshape((nAct * nAgents, nInit, nSim, windowSize))
    return np.all(((np.max(C, axis=3) - np.min(C, axis=3))/np.min(C, axis=3)) < tol, axis=0)

def pLCE(data):
    data = np.reshape(data.transpose(1, 3, 0, 2, 4), (nAgents * nAct, nInit, nSim, nIter))
    return np.array([np.mean(np.linalg.norm(data[:, 1:, :, i] - np.expand_dims(data[:, 0, :, i], axis=1), axis=0), axis=0) for i in range(nIter)])

class Agent:
    def __init__(self, number, nAct, nAgents, G, L):
        self.number = number
        self.nAct = nAct
        self.nAgents = nAgents
        self.opponents = list(range(nAgents))
        self.opponents.remove(self.number)
        B = np.array([G[L[self.number, o]] for o in self.opponents])
        self.payoffs = np.dstack([block_diag(*[B[i, :, :, s] for i in range(nAgents - 1)]) for s in range(nSim)])
        self.P = np.zeros(nAct)

    def getP(self, x, nInit):
        opState = x[self.opponents].transpose(0, 3, 1, 2).reshape((self.nAct * len(self.opponents), nInit, nSim))
        k = np.einsum('kjs,jsi->ksi', self.payoffs, opState.transpose(0, 2, 1))
        return np.sum(k.reshape(nAgents - 1, nAct, nInit, nSim), axis=0).transpose(1, 2, 0)

if __name__ == "__main__":
    nAct = 35
    nAgents = 2
    nInit = 10
    nSim = 25
    nNbr = nInit - 1
    tryno = 2

    beta = 5e-2
    nIter = int(5e4)
    numTests = 20

    windowSize = int(1e4)
    
    L = np.zeros((nAgents, nAgents), dtype=int)
    for i in range(nAgents):
        for j in range(i):
            L[i, j] = 1

    allConv = np.zeros((numTests, numTests))

    x = np.random.dirichlet(np.ones(nAct), size=(nAgents, nInit, nSim))

    for i, gamma in tqdm(enumerate(np.linspace(-1, 0, num=numTests))):
        C, D = generateGames(gamma, nSim, nAct)
        for j, alpha in enumerate(np.linspace(0.01, 0.03, num=numTests)):
            allConverged = 0
            # allActions = np.zeros((nAgents, nInit, nSim, nAct, nIter))

            G = (C, np.transpose(D, (1, 0, 2)))
            agents = [Agent(i, nAct, nAgents, G, L) for i in range(nAgents)]
            checkWindow = np.zeros((nAgents, nInit, nSim, nAct, windowSize))
            n = 0
            for cIter in range(nIter):
                P = np.stack([agents[i].getP(x, nInit) for i in range(nAgents)])
                x = ((x ** (1 - alpha)) * np.exp(beta * P)) / np.sum((x ** (1 - alpha)) * np.exp(beta * P),
                                                                             axis=3)[:, :, :, None]
                # allActions[:, :, :, :, cIter] = x

                # allActions = allActions.transpose(1, 0, 2, 3, 4)
            
            
                if cIter > (nIter - windowSize - 1):
                    checkWindow[:, :, :, :, n] = x
                    n += 1
        
            converged = checkMinMax(checkWindow, windowSize, 1e-2)

            allConv[numTests - 1 - i, j] = np.sum(converged)/(nInit * nSim)

    np.savetxt('p{0}n{1}trial{2}.csv'.format(nAgents, nAct, tryno), allConv)

    sns.heatmap(allConv), plt.show()