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

    wx = np.random.rand()
    wy = np.random.rand()
    wz = np.random.rand()
    W = np.array([[0, wx, 1 - wx], [wy, 0, 1 - wy], [wz, 1 - wz, 0]])

    beta = np.random.rand()
    B = np.array([[-beta, 1, 0], [0, -beta, 1], [1, 0, -beta]])
    A = -W[1, 0] * B.T
    C = -W[1, 2] * B
    G = np.dstack((A, B, C))

    strats = fpn.simulate(dim, G, W, 3, nInit=nInit, nIter=1e5, noise=False, sigma=.275)
    plotOnSimplex(strats, 3, nInit)