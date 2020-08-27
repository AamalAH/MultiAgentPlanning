import numpy as np
import itertools
import scipy.sparse as sps
import matplotlib.pyplot as plt
from tqdm import tqdm

gamma = .1
tau = 10

nSim = 10
nIter = int(1e4)
nTests = 10

nPlayers = 3
nActions = 2

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

    allPayoffs = np.zeros(shape=(nPlayers, nElements))

    for i in range(nSim):
        rewards = np.random.multivariate_normal(np.zeros(nPlayers * nElements), cov=cov)
        each = np.array_split(rewards, nPlayers)
        allPayoffs = np.dstack((allPayoffs, np.array(each)))

    return allPayoffs

def getActionProbs(qValues):
    partitionFunction = np.expand_dims(np.sum(np.exp(tau * qValues), axis=1), axis = 1)
    partitionFunction = np.hstack([partitionFunction]*nActions)
    actionProbs = np.exp(tau * qValues)/partitionFunction
    return actionProbs


def qUpdate(qValues, payoffs):
    idX = list(itertools.product(list(range(0, nActions)), repeat=nPlayers))
    actionProbs = getActionProbs(qValues)
    bChoice = np.array([[np.random.choice(list(range(nActions)), p=actionProbs[p, :, s]) for s in range(nSim)] for p in range(nPlayers)])

    rewards = np.array([[payoffs[p, np.where([np.all(idX[i] == np.roll(bChoice[:, s], -1 * p)) for i in range(len(idX))])[0][0], s] for s in range(nSim)] for p in range(nPlayers)])
    for s in range(nSim):
        qValues[range(nPlayers), bChoice[:, s], s] += alpha * (rewards[:, s] - qValues[range(nPlayers), bChoice[:, s], s] + gamma * np.max(qValues[:, :, s], axis = 1))

    return qValues

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

  normSteps = np.array([[np.linalg.norm((x - xOld)[p, :, i]) for i in range(nSim)] for p in range(nPlayers)])
  return normSteps < tol

plotConv = []

x = np.zeros((nPlayers, nActions, nSim))
for alpha in tqdm(np.linspace(0, 1, num=5)):
    for Gamma in tqdm(np.linspace(-1, 1, num=5)):

        payoffs = generateGames(Gamma, nSim, nActions, nPlayers)

        qValues0 = np.random.rand(nPlayers, nActions, nSim)

        stopCond = False
        for cIter in range(nIter):
            qValues0 = qUpdate(qValues0, payoffs)
            
            xOld = np.copy(x)
            x = getActionProbs(qValues0)
            stopCond = checkConvergence(x, xOld, 1e-3)
            if np.all(stopCond):
                break

        plotConv += [np.array([alpha, Gamma, np.mean(np.all(stopCond, axis = 0))])]

       
        