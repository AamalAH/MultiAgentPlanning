import numpy as np
from scipy.linalg import block_diag
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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
    C = checkWindow.reshape((nAct * nAgents, nInit, windowSize))
    return np.all(((np.max(C, axis=2) - np.min(C, axis=2))/np.min(C, axis=2)) < tol, axis=0)

class Agent:
    def __init__(self, number, nAct, nAgents, G, L):
        self.number = number
        self.nAct = nAct
        self.nAgents = nAgents
        self.opponents = list(range(nAgents))
        self.opponents.remove(self.number)
        self.payoffs = block_diag(*[G[L[1, o]] for o in self.opponents])
        self.P = np.zeros(nAct)

    def getP(self, x, nInit):
        k = self.payoffs @ x[self.opponents].transpose(0, 2, 1).reshape((self.nAct * (self.nAgents - 1), nInit))
        return np.sum(k.reshape(nAgents - 1, nAct, nInit), axis=0).T

if __name__ == "__main__":
    nAct = 35
    nAgents = 2
    nInit = 5
    nSim = 20

    beta = 5e-2
    nIter = int(5e4)
    numTests = 20
    allConv = np.zeros((numTests, numTests))

    windowSize = int(1e4)

    L = np.zeros((nAgents, nAgents), dtype=int)
    for i in range(nAgents):
        for j in range(i):
            L[i, j] = 1

    allConv = np.zeros((numTests, numTests))

    x = np.random.dirichlet(np.ones(nAct), size=(nAgents, nInit))
    for i, gamma in tqdm(enumerate(np.linspace(-1, 0, num=numTests))):
        C, D = generateGames(gamma, nSim, nAct)
        for j, alpha in enumerate(np.linspace(0.0, 0.03, num=numTests)):
            allConverged = 0
            for cSim in range(nSim):
                A, B = C[:, :, cSim], D[:, :, cSim]
                G = (A, B.T)
                agents = [Agent(i, nAct, nAgents, G, L) for i in range(nAgents)]
                checkWindow = np.zeros((nAgents, nInit, nAct, windowSize))
                n = 0
                for cIter in range(nIter):
                    P = np.stack([agents[i].getP(x, nInit) for i in range(nAgents)])
                    x = ((x ** (1 - alpha)) * np.exp(beta * P)) / np.sum((x ** (1 - alpha)) * np.exp(beta * P),
                                                                                 axis=2)[:, :, None]
                    if cIter > (nIter - windowSize - 1):
                        checkWindow[:, :, :, n] = x
                        n += 1

                converged = checkMinMax(checkWindow, windowSize, 1e-2)

                allConverged += np.sum(converged)

            allConv[numTests - 1 - i, j] = allConverged/(nInit * nSim)

    sns.heatmap(allConv), plt.show()