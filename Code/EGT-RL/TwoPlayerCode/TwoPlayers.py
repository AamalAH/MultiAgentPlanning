from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

def generateGame(gamma, nAct):
    reward1s = {}
    reward2s = {}

    nElements = nAct**2 #number of payoff elements in the matrix

    cov = np.eye(2 *  nElements)
    cov[:nElements, nElements:] = np.eye(nElements) * gamma
    cov[nElements:, :nElements] = np.eye(nElements) * gamma

    rewards = np.random.multivariate_normal(np.zeros(2 * nElements), cov=cov)

    reward1s = rewards[0:nElements].reshape((nAct, nAct))
    reward2s = rewards[nElements:].reshape((nAct, nAct)).T

    return [reward1s, reward2s]

def checkConvergence(actionProb, oldxBar, tol):
    normStep = np.linalg.norm(actionProb - oldxBar)
    return normStep < tol

def runSim(params, nSim, tol):

    alpha = params[0]
    tau = params[1]
    gamma = params[2]
    Gamma = params[3]

    xsAll = []
    converged = []

    for cSim in range(nSim):
        stopCond = False

        [payoffA, payoffB] = generateGame(Gamma, 2)

        qValues = np.random.rand(2, 2)
        xs = []
        oldxBar = np.array([1, 1])
        for cIter in range(int(5e5)):

            partitionFunction = [np.sum([np.exp(tau * i) for i in qValues[p, :]]) for p in range(2)]
            actionProbs = np.array([[np.exp(tau * i)/partitionFunction[p] for i in qValues[p, :]] for p in range(2)])

            xs += [actionProbs[:, 0]]

            if cIter % (int(1e4)) == 0 and cIter != 0:
                stopCond = checkConvergence(actionProbs[:, 0], oldxBar, tol)
                if stopCond:
                    break
                oldxBar = actionProbs[:, 0]

            boltzmannChoices = [np.random.choice([0, 1], p=actionProbs[p, :]) for p in range(2)]

            rewards = [payoffA[boltzmannChoices[0], boltzmannChoices[1]], payoffB[boltzmannChoices[0], boltzmannChoices[1]]]

            qValues[0, boltzmannChoices[0]] += alpha * (rewards[0] - qValues[0, boltzmannChoices[0]] + gamma * max(qValues[0, :]))
            qValues[1, boltzmannChoices[1]] += alpha * (rewards[1] - qValues[1, boltzmannChoices[1]] + gamma * max(qValues[1, :]))

        xs = np.array(xs)

        xsAll += [xs]
        converged += [stopCond]

    return xsAll, np.mean(converged)


if __name__ == "__main__":

    dirName = 'ParameterSweep Results'
    if os.path.exists(dirName):
        shutil.rmtree(dirName)

    if not os.path.exists(dirName):
        os.makedirs(dirName)

    alpha = .1
    gamma = .1
    Gamma = 0

    numTests = 100

    allmeanConvergence = np.zeros((numTests, numTests))

    i = 0
    for tau in tqdm(np.linspace(0.1, 10, num=numTests)):
        j = 1
        for Gamma in np.linspace(-1, 1, num=numTests):

            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.set_xlim([0, 1])
            # ax.set_ylim([0, 1])
            xsAll, convergenceRate = runSim([alpha, tau, gamma, Gamma], 20, 1e-2)

            with open(dirName + '/parameterSweep_tau_{0}_gamma_{1}.txt'.format(tau, Gamma), 'w') as f:
                f.write('tau: {0}, gamma: {1} \n'.format(tau, Gamma))
                f.write('Converged: {0} \n'.format(str(convergenceRate)))

                f.close()

            # for X in xsAll:
            #     ax.plot(X[:, 0], X[:, 1], 'k--', zorder=1)
            #     ax.scatter(X[0, 0], X[0, 1], color= 'y', marker='.')
            #
            # for X in xsAll:
            #     ax.scatter(X[-1, 0], X[-1, 1], color='r', marker='+', zorder=2)
            #
            # fig.savefig(dirName + '/parameterSweep_tau_{0}_gamma_{1}.png'.format(tau, Gamma))
            # plt.close(fig)
            allmeanConvergence[numTests - j, i] = convergenceRate

            j += 1
        i += 1