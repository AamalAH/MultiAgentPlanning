from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil


def generateGame(gamma, nAct):
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

    rewards = np.random.multivariate_normal(np.zeros(2 * nElements), cov=cov)

    reward1s = rewards[0:nElements].reshape((nAct, nAct))
    reward2s = rewards[nElements:].reshape((nAct, nAct)).T

    return [reward1s, reward2s]

def checkConvergence(actionProb, oldxBar, tol):
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
    normStep1 = np.linalg.norm(actionProb[0, :] - oldxBar[0, :])
    normStep2 = np.linalg.norm(actionProb[1, :] - oldxBar[1, :])
    return (normStep1 < tol) and (normStep2 < tol)

def runSim(params, nSim, tol):
    alpha = params[0]
    tau = params[1]
    gamma = params[2]
    Gamma = params[3]

    xsAll = []
    converged = []

    for cSim in range(nSim):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        stopCond = False

        [payoffA, payoffB] = generateGame(Gamma, 2)

        payoffA, payoffB = np.array([[1, 5], [0, 3]]), np.array([[1, 0], [5, 3]])

        qValues = np.random.rand(2, 2)
        xs = []
        oldxBar = np.ones((2, 2))
        for cIter in range(int(1e5)):

            partitionFunction = [np.sum([np.exp(tau * i) for i in qValues[p, :]]) for p in range(2)]
            actionProbs = np.array([[np.exp(tau * i) / partitionFunction[p] for i in qValues[p, :]] for p in range(2)])

            xs += [actionProbs[:, 0]]

            # if cIter % (int(1e4)) == 0 and cIter != 0:
            #     stopCond = checkConvergence(actionProbs, oldxBar, tol)
            #     if stopCond:
            #         break
            #     oldxBar = actionProbs

            boltzmannChoices = [np.random.choice([0, 1], p=actionProbs[p, :]) for p in range(2)]

            rewards = [payoffA[boltzmannChoices[0], boltzmannChoices[1]],
                       payoffB[boltzmannChoices[0], boltzmannChoices[1]]]

            qValues[0, boltzmannChoices[0]] += alpha * (
                        rewards[0] - qValues[0, boltzmannChoices[0]] + gamma * max(qValues[0, :]))
            qValues[1, boltzmannChoices[1]] += alpha * (
                        rewards[1] - qValues[1, boltzmannChoices[1]] + gamma * max(qValues[1, :]))

        xs = np.array(xs)

        xsAll += [xs]
        converged += [stopCond]

    for X in xsAll:
        ax.plot(X[:, 0], X[:, 1], 'k--', zorder=1)
        ax.scatter(X[0, 0], X[0, 1], color= 'y', marker='.')

    for X in xsAll:
        ax.scatter(X[-1, 0], X[-1, 1], color='r', marker='+', zorder=2)

    plt.show()
    print('hi')

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
    tau = 0.5

    numTests = 10

    allmeanConvergence = np.zeros((numTests, numTests))

    i = 0
    for alpha in tqdm(np.linspace(0, 1, num=numTests)):
        j = 1
        for Gamma in np.linspace(-1, 1, num=numTests):

            if not os.path.exists(dirName + '/parameterSweep_alpha_{0}_gamma_{1}.txt'.format(alpha, Gamma)):

                xsAll, convergenceRate = runSim([0.8, tau, gamma, -0.33], 10, 1e-2)

                with open(dirName + '/parameterSweep_alpha_{0}_gamma_{1}.txt'.format(alpha, Gamma), 'w') as f:
                    f.write('tau: {0}, gamma: {1} \n'.format(alpha, Gamma))
                    f.write('Converged: {0} \n'.format(str(convergenceRate)))

                    f.close()

                # fig.savefig(dirName + '/parameterSweep_tau_{0}_gamma_{1}.png'.format(tau, Gamma))
                # plt.close(fig)
                allmeanConvergence[numTests - j, i] = convergenceRate

            j += 1
        i += 1