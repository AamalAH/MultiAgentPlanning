import numpy as np
import matplotlib.pyplot as plt
import FPNAGFunctions as fpn

def plotOnSimplex(traj, nAgents, nInit):
    for cAgent in range(nAgents):
        f, ax = plt.subplots(1, 1, sharex='all', sharey='all')

        proj = np.array(
            [[-1 * np.cos(30 / 360 * 2 * np.pi), np.cos(30 / 360 * 2 * np.pi), 0],
             [-1 * np.sin(30 / 360 * 2 * np.pi), -1 * np.sin(30 / 360 * 2 * np.pi), 1]
             ])

        ts = np.linspace(0, 1, 10000)

        e1 = proj @ np.array([ts, 1 - ts, 0 * ts])
        e2 = proj @ np.array([0 * ts, ts, 1 - ts])
        e3 = proj @ np.array([ts, 0 * ts, 1 - ts])

        ax.plot(e1[0], e1[1], 'k')
        ax.plot(e2[0], e2[1], 'k')
        ax.plot(e3[0], e3[1], 'k')

        for i in range(nInit):
            d = proj @ traj[:, cAgent, i, :]
            ax.plot(d[0], d[1], '--', alpha=0.6)
            ax.scatter(d[0, -1], d[1, -1], color='r', marker='+')

        plt.show()

if __name__ == "__main__":
    dim = 3
    nAgents = 10
    nInit = 10
    W = fpn.initialiseAggregationMatrix(nAgents=10)

    G = fpn.generateNetworkGame(dim, nAgents)

    strats = fpn.simulate(dim, G, W, nAgents, nInit=nInit, nIter=1e5, noise=False, sigma=1.5)

    plotOnSimplex(strats, nAgents, nInit)

