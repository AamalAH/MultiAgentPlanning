import numpy as np
from scipy.integrate import odeint
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

def getActionProbs(Q, agentParams):
    """
        qValues: nPlayer x nActions x nSim
        return: nPlayer x nActions x nSim
        """
    alpha, tau = agentParams

    return np.exp(tau * Q) / np.sum(np.exp(tau * Q), axis=1)[:, None]

def TuylsODE(X, t, G, agentParams):
    alpha, beta = agentParams
    A, B = G
    x, y, z = X[0:nAct], X[nAct:2*nAct], X[2*nAct:]
    xdot, ydot, zdot = np.zeros(nAct), np.zeros(nAct), np.zeros(nAct)
    for i in range(nAct):
        xdot[i] = x[i] * alpha * beta * (((A @ y)[i] + (B.T @ z)[i] - np.dot(x, A @ y + B.T @ z)) - (1/beta) * (np.log(x[i]) - np.dot(x, np.log(x))))
        ydot[i] = y[i] * alpha * beta * (((A @ z)[i] + (B.T @ x)[i] - np.dot(y, A @ z + B.T @ x)) - (1/beta) * (np.log(y[i]) - np.dot(y, np.log(y))))
        zdot[i] = z[i] * alpha * beta * (((A @ x)[i] + (B.T @ y)[i] - np.dot(z, A @ x + B.T @ y)) - (1/beta) * (np.log(z[i]) - np.dot(z, np.log(z))))

    return np.hstack((xdot, ydot, zdot))

def checkminMax(checkWindow, windowSize, tol):
    C = checkWindow.reshape((nAct * nAgents, windowSize))
    relDiff = (np.max(C, axis=1) - np.min(C, axis=1)) / np.min(C, axis=1)
    return np.all(relDiff < tol)

# def checkminMaxInit(checkWindow, windowSize, tol):
#     C = checkWindow.reshape((nAct * nAgents, nInit, windowSize))
#     relDiff = (np.max(C, axis=2) - np.min(C, axis=2)) / np.min(C, axis=2)
#     return np.all(relDiff < tol, axis=0)

def checkVar(checkWindow, windowSize, tol):
    C = checkWindow.reshape((nAct * nAgents, nInit, windowSize))
    return np.mean(np.var(C, axis=2), axis=0) < tol

nAct = 2
nAgents = 3
alpha, beta = 1e-2, 5e-2
agentParams = (alpha, beta)
nIter = int(1.5e4)
nInit = 15
nSim = 5
checkStart = int(5e3)
windowSize = int(5e3)

x = np.random.dirichlet(np.ones(nAct), size=(nAgents))
x0 = np.copy(x).reshape(nAct * nAgents)
allX = np.zeros((nAgents, nAct, nIter))
allXQ = np.zeros((nAgents, nAct, nIter))

A, B = generateGames(-1, 1, nAct)
A, B = A.squeeze(), B.squeeze()
G = (A, B)
# checkWindow = np.zeros((nAgents, nAct, windowSize))
# n = 0
# converged = False
#
# for cIter in range(nIter):
#     P = np.vstack((A @ x[1] + B.T @ x[2], A @ x[2] + B.T @ x[0], A @ x[0] + B.T @ x[1]))
#     x = ((x**(1 - alpha)) * np.exp(alpha * beta * P)) / np.sum((x**(1 - alpha)) * np.exp(alpha * beta * P), axis=1)[:, None]
#     allX[:, :, cIter] = x
#
# t = np.linspace(0, nIter, 10*nIter + 1)
# sol = odeint(TuylsODE, x0, t, args=(G, agentParams))
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 1])
# ax.set_zlim([0, 1])

# ax.plot(sol[:, 0], sol[:, 2], sol[:, 4], color = 'r')
# ax.scatter(sol[0, 0], sol[0, 2], sol[0, 4], marker='o', color='r')
# ax.scatter(sol[-1, 0], sol[-1, 2], sol[-1, 4], marker='+', color='k')

# allX = allX.reshape(6, nIter)

# ax.plot(allX[0, :], allX[2, :], allX[4, :], color='y')
# ax.scatter(allX[0, 0], allX[2, 0], allX[4, 0], marker='o', color='r')
# ax.scatter(allX[0, -1], allX[2, -1], allX[4, -1], marker='+', color='k')

# plt.show()
# allConverged = 0
#
# for cSim in range(nSim):
#     x = np.random.dirichlet(np.ones(nAct), size=(nAgents, nInit))
#     checkWindow = np.zeros((nAgents, nInit, nAct, windowSize))
#     n = 0
#     converged = False
#     for cIter in range(nIter):
#         P = np.stack((np.einsum('ij, nj -> ni', A[:, :, cSim], x[1]) + np.einsum('ij, nj -> ni', B[:, :, cSim].T, x[2]), np.einsum('ij, nj -> ni', A[:, :, cSim], x[2]) + np.einsum('ij, nj -> ni', B[:, :, cSim].T, x[0]), np.einsum('ij, nj -> ni', A[:, :, cSim], x[0]) + np.einsum('ij, nj -> ni', B[:, :, cSim].T, x[1])))
#         x = ((x ** (1 - alpha)) * np.exp(alpha * beta * P)) / np.sum((x ** (1 - alpha)) * np.exp(alpha * beta * P), axis=2)[:, :, None]
#         if cIter > checkStart:
#             if n < windowSize:
#                 checkWindow[:, :, :, n] = x
#                 n += 1
#             elif n == windowSize:
#                 converged = checkminMaxInit(checkWindow, windowSize, 1e-2)
#                 if np.all(converged):
#                     break
#             else:
#                 checkWindow = np.zeros((nAgents, nAct, windowSize))
#                 n = 0
#     allConverged += np.sum(converged)
#
# convRate = allConverged/(nInit * nSim)

# Q = np.random.rand(nAgents, nAct)
# checkWindow = np.zeros((nAgents, nAct, windowSize))
# n=0
# convergedQ = False
#
# for cIter in range(nIter):
#     xq = getActionProbs(Q, agentParams)
#     bChoice = np.array([np.random.choice(range(nAct), p=xq[p]) for p in range(nAgents)])
#     rewardA = A[bChoice[0], bChoice[1]] + B[bChoice[2], bChoice[0]]
#     rewardB = B[bChoice[0], bChoice[1]] + A[bChoice[1], bChoice[2]]
#     rewardC = A[bChoice[2], bChoice[0]] + B[bChoice[1], bChoice[2]]
#
#     Q[0, bChoice[0]] += alpha * (rewardA - Q[0, bChoice[0]])
#     Q[1, bChoice[1]] += alpha * (rewardB - Q[1, bChoice[1]])
#     Q[2, bChoice[2]] += alpha * (rewardC - Q[2, bChoice[2]])
#
#     allXQ[:, :, cIter] = xq
#
#     if cIter > checkStart:
#         if n < windowSize:
#             checkWindow[:, :, n] = xq
#             n += 1
#         elif n == windowSize:
#             convergedQ = checkminMax(checkWindow, windowSize, 1e-4)
#             if convergedQ:
#                 break
#         else:
#             checkWindow = np.zeros((nAgents, nAct, windowSize))
#             n = 0

nAct = 25
nAgents = 3
numTests = 10
beta = 0.05
allConv = np.zeros((numTests, numTests))

for i, gamma in tqdm(enumerate(np.linspace(-1, 0, num=numTests))):
    for j, alpha in enumerate(np.linspace(0.01, 0.03, num=numTests)):
        allConverged = 0
        A, B = generateGames(gamma, nSim, nAct)
        for cSim in range(nSim):
            x = np.random.dirichlet(np.ones(nAct), size=(nAgents, nInit))
            checkWindow = np.zeros((nAgents, nInit, nAct, windowSize))
            n = 0
            for cIter in range(nIter):
                P = np.stack((np.einsum('ij, nj -> ni', A[:, :, cSim], x[1]) + np.einsum('ij, nj -> ni', B[:, :, cSim].T, x[2]), np.einsum('ij, nj -> ni', A[:, :, cSim], x[2]) + np.einsum('ij, nj -> ni', B[:, :, cSim].T, x[0]), np.einsum('ij, nj -> ni', A[:, :, cSim], x[0]) + np.einsum('ij, nj -> ni', B[:, :, cSim].T, x[1])))
                x = ((x ** (1 - alpha)) * np.exp(alpha * beta * P)) / np.sum((x ** (1 - alpha)) * np.exp(alpha * beta * P), axis=2)[:, :, None]
                if cIter > (nIter - windowSize - 1):
                    checkWindow[:, :, :, n] = x
                    n += 1

            converged = checkVar(checkWindow, windowSize, 1e-5)
                # if cIter >= nIter - windowSize:
                #     if n < windowSize:
                #         checkWindow[:, :, :, n] = x
                #         n += 1
                #     elif n == windowSize:
                #         converged = checkVar(checkWindow, windowSize, 1e-5)
                #         if np.all(converged):
                #             break
                #     else:
                #         checkWindow = np.zeros((nAgents, nAct, windowSize))
                #         n = 0
            allConverged += np.sum(converged)

        allConv[numTests - 1 - i, j] = allConverged/(nInit * nSim)
sns.heatmap(allConv), plt.show()


