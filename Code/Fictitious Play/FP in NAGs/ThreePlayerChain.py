import numpy as np
import matplotlib.pyplot as plt
import FPNAGFunctions as fpn

def plotOnSimplex(traj, nAgents, nInit):
    f, ax = plt.subplots(1, nAgents, sharex='all', sharey='all')

    proj = np.array(
        [[-1 * np.cos(30 / 360 * 2 * np.pi), np.cos(30 / 360 * 2 * np.pi), 0],
         [-1 * np.sin(30 / 360 * 2 * np.pi), -1 * np.sin(30 / 360 * 2 * np.pi), 1]
         ])

    ts = np.linspace(0, 1, 10000)

    e1 = proj @ np.array([ts, 1 - ts, 0 * ts])
    e2 = proj @ np.array([0 * ts, ts, 1 - ts])
    e3 = proj @ np.array([ts, 0 * ts, 1 - ts])

    for cAgent in range(nAgents):

        ax[cAgent].plot(e1[0], e1[1], 'k')
        ax[cAgent].plot(e2[0], e2[1], 'k')
        ax[cAgent].plot(e3[0], e3[1], 'k')

        for i in range(nInit):
            d = proj @ traj[:, cAgent, i, :]
            ax[cAgent].plot(d[0], d[1], '--', alpha=0.4)
        for i in range(nInit):
            ax[cAgent].scatter(d[0, -1], d[1, -1], color='k', marker='+')

    plt.show()

if __name__ == "__main__":
    dim = 3
    nInit = 5

    # Normal Three Player Chain
    w = 0.2882028455979141
    W = np.array([[0, 1, 0], [w, 0, 1-w], [0, 1, 0]])

    # Generic Aggregation Matrix
    # W = fpn.initialiseAggregationMatrix(3)

    # Three Player Chain without the diagonal condition

    # w1, w2, w3 = np.random.rand(), np.random.rand(), np.random.rand()
    # W = np.array([[w1, 1 - w1, 0], [w2, 0, 1 - w2], [0, w3, 1 - w3]])

    # G = fpn.generateThreePlayerChain(dim, W)

    # Shapley Orbit
    beta = 0.5756078897829536
    B = np.array([[-beta, 1, 0], [0, -beta, 1], [1, 0, -beta]])
    A = -W[1, 0] * B.T
    C = -W[1, 2] * B.T
    G = np.dstack((A, B, C))

    strats = fpn.simulate(dim, G, W, 3, nInit=nInit, nIter=1e5, noise=True, sigma=.1)
    plotOnSimplex(strats, 3, nInit)