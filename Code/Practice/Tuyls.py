import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

        # rewardAs = np.dstack((rewardAs, rewards[0:nElements].reshape((nAct, nAct))))
        rewardAs = np.dstack((rewardAs, np.array([[1, 5], [0, 3]])))
        # rewardBs = np.dstack((rewardBs, rewards[nElements:].reshape((nAct, nAct)).T))
        rewardBs = np.dstack((rewardBs, np.array([[1, 3], [0, 5]]).T))
    return [rewardAs[:, :, 1:], rewardBs[:, :, 1:]]

alpha = 2e-2
tau = 10

x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)

X, Y = np.meshgrid(x, y)

A, B = np.array(([[-1.08563096,  1.7272803],   [0.0719464,   0.22215408]])), np.array(([[-0.04231243,  0.57071329],  [1.0930919,  -1.68208477]]))

E = lambda R: (R * np.log(R/R)) + ((1 - R) * np.log((1 - R)/R))

x_dot = lambda M, N: alpha * M * tau * ((A @ [N, 1-N])[0] - np.dot([M, 1-M], A@[N, 1-N])) + M * alpha * E(M)

y_dot = lambda M, N: alpha * N * tau * ((B @ [M, 1-M])[0] - np.dot([N, 1-N], B@[M, 1-M])) + N * alpha * E(N)

U, V = np.zeros(X.shape), np.zeros(X.shape)
NI, NJ = X.shape

for i in range(NI):
    for j in range(NJ):
        U[i, j] = x_dot(X[i, j], Y[i, j])
        V[i, j] = y_dot(X[i, j], Y[i, j])

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.quiver(X, Y, U, V, color = 'b', width=2e-3)
# plt.savefig('alpha_{0}_gamma_{1}'.format(int(alpha * 1e2), int(Gamma)))
plt.show()
