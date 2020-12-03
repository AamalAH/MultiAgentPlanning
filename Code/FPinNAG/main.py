import numpy as np
import matplotlib.pyplot as plt

"""
DO in NAG simulation: 

1. Set up adjacency matrix
2. Create function for cost function
3. Initialise agent preferences
4. Set up initial references

Until convergence:
Each agent simultaneously:
Computes optimal strategy
Communicates the new optimal strategy to the neighbours
"""

def initialise():
    """
    Initialise all required components for the NAG problem in Parise et al.
    :return: (NxN) Adjacency Matrix, (Nx1) q array with all entries > 1, (nxN) x0 array, (nxN) sigma array
    """

    adjMat = (np.eye(3) == False) * (1 / 2)  # Designed to conform to double stochastic property

    initVectors = np.random.rand(2, 3)
    initVectors /= initVectors.sum(axis=0)

    initQ = np.random.randint(2, 5, (3,))  # Has to be higher than 1 but upper bound is arbitrary

    initRef = (adjMat @ initVectors.T).T
    return adjMat, initQ, initVectors, initRef

def bestResponse(i, sigmaRef, Q, xPreF):
    xPreF = np.expand_dims(xPreF[:, i], axis=1)
    APV = np.vstack((np.linspace(0, 1, 50), 1 - np.linspace(0, 1, 50)))
    term1 = Q[i] * np.linalg.norm(np.subtract(APV, xPreF), axis=0) ** 2
    term2 = 2 * sigmaRef[:, i].T @ APV

    bestCost = np.min(term1 + term2)
    bestReply = np.argmin(term1 + term2)

    return APV[:, bestReply], bestCost

def updateSigma(adjMat, vectors):
    return (adjMat @ vectors.T).T

def addNoise(prob):
    APV = np.vstack((np.linspace(0, 1, 50), 1 - np.linspace(0, 1, 50)))
    approxIdx = np.argmin(np.linalg.norm(APV - np.expand_dims(prob, axis=1), axis=0))
    noisyIdx = np.random.randint(-2, 2)
    while not (approxIdx + noisyIdx >= len(APV) or approxIdx + noisyIdx < 0):
        noisyIdx = np.random.randint(-2, 2)
    return APV[:, approxIdx + noisyIdx]

def EKF(z, x, P, Q, K, R):
    Pf = P + Q
    x += K(z - x)
    Kk = Pf @ np.linalg.inv(Pf + R)
    P = (np.eye(2) - Kk) @ Pf

    return

def KF(z, x, P):

    Q = np.eye(4) * 0.05
    R = np.eye(4) * 0.02
    H = np.eye(4)

    P += Q
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    dz = z - x

    x += K @ dz
    P = (np.eye(4) - K @ H) @ P

    return x.reshape((2, 2)), P

def createNoisyData(data):
    noisyData = [np.vstack((data[:, 0], addNoise(data[:, 1]), addNoise(data[:, 2]))).T]
    noisyData += [np.vstack((addNoise(data[:, 0]), data[:, 1], addNoise(data[:, 2]))).T]
    noisyData += [np.vstack((addNoise(data[:, 0]), addNoise(data[:, 1]), data[:, 2])).T]
    return noisyData

def createEstimates(data, currentEstimates, allP):
    for i in range(3):
        idx = list(range(3))
        idx.remove(i)
        trueIdx = [i] + idx
        estimates, allP[i] = KF(data[i][:, idx].reshape((4)), currentEstimates[i][:, idx].reshape((4)), allP[i])
        pieces = [currentEstimates[i][:, i]] + [estimates[:, j] for j in range(2)]
        currentEstimates[i] = np.array([pieces[j] for j in trueIdx]).T

    return currentEstimates, allP

def noisyNAG(adjMat, initQ, initVectors, initRef, xPreF=None, process="Parise"):
    Q = initQ
    sigmaRef = initRef
    probVectors = initVectors

    cIter = 0
    stopCond = False

    allProbs = []
    allSigs = []
    allMeanCost = []

    allP = [np.eye(4) * 0.1] * 3
    noisyData = createNoisyData(probVectors)

    currentEstimates = createNoisyData(probVectors)

    allSigma = [updateSigma(adjMat, currentEstimates[0]),
                updateSigma(adjMat, currentEstimates[1]),
                updateSigma(adjMat, currentEstimates[2])]
    allProbs = [[], [], []]

    while not stopCond:
        [allProbs[i].append(currentEstimates[i]) for i in range(3)]

        probVectors = np.array([bestResponse(i, allSigma[i], Q, xPreF)[0] for i in range(3)]).T

        averageProbs = [(1 / len(allProbs[i])) * np.sum(np.array(allProbs[i]), axis=0) for i in range(3)]

        currentEstimates, allP = createEstimates(createNoisyData(probVectors), currentEstimates, allP)

        if process == "Parise":
            allSigma = [updateSigma(adjMat, currentEstimates[0]),
                        updateSigma(adjMat, currentEstimates[1]),
                        updateSigma(adjMat, currentEstimates[2])]
        elif process == "FP":
            allSigma = [updateSigma(adjMat, averageProbs[0]),
                        updateSigma(adjMat, averageProbs[1]),
                        updateSigma(adjMat, averageProbs[2])]
        else:
            raise Exception("An invalid procedure was entered")

        cIter += 1
        stopCond = cIter >= 100

    return np.dstack((np.array(allProbs[0])[:, :, 0], np.array(allProbs[1])[:, :, 1], np.array(allProbs[2])[:, :, 2]))

def NAGProcedure(adjMat, initQ, initVectors, initRef, xPreF=None, process="Parise"):
    Q = initQ
    sigmaRef = initRef
    probVectors = initVectors

    cIter = 0
    stopCond = False

    allProbs = []

    while not stopCond:
        allProbs += [probVectors]

        probVectors = np.array([bestResponse(i, sigmaRef, Q, xPreF)[0] for i in range(3)]).T

        averageProbs = (1 / len(allProbs)) * np.sum(np.array(allProbs), axis=0)

        if process == "Parise":
            sigmaRef = updateSigma(adjMat, probVectors)
        elif process == "FP":
            sigmaRef = updateSigma(adjMat, averageProbs)
        else:
            raise Exception("An invalid procedure was entered")

        cIter += 1
        stopCond = cIter >= 100

    return np.array(allProbs)


def getDistFromDesX(X, desX):
    return np.linalg.norm((X.T - np.expand_dims(desX[:, 0], axis=1)), axis=0)


def plotNAGConvergence(desX, *args):
    fig = plt.figure()

    ax = fig.add_subplot(131)
    ax.plot(getDistFromDesX(args[0][:, :, 0], desX), 'b-')
    ax.plot(getDistFromDesX(args[1][:, :, 0], desX), 'r--')
    ax.set_title('Player 1', y=-0.15)

    ax = fig.add_subplot(132)
    ax.plot(getDistFromDesX(args[0][:, :, 1], desX), 'b-')
    ax.plot(getDistFromDesX(args[1][:, :, 1], desX), 'r--')
    ax.set_title('Player 2', y=-0.15)

    ax = fig.add_subplot(133)
    ax.plot(getDistFromDesX(args[0][:, :, 2], desX), 'b-', label="Parise")
    ax.plot(getDistFromDesX(args[1][:, :, 2], desX), 'r--', label="FP")
    ax.set_title('Player 3', y=-0.15)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.show()


if __name__ == "__main__":
    (adjMat, initQ, initVectors, initRef) = initialise()

    xPreF = np.random.rand(2, 3)
    xPreF /= xPreF.sum(axis=0)

    allProbsNoisyParise = noisyNAG(adjMat, initQ, initVectors, initRef, xPreF, process="Parise")
    allProbsNoisyFP = noisyNAG(adjMat, initQ, initVectors, initRef, xPreF, process="FP")

    allProbsParise = NAGProcedure(adjMat, initQ, initVectors, initRef, xPreF, process="Parise")
    allProbsFP = NAGProcedure(adjMat, initQ, initVectors, initRef, xPreF, process="FP")

    plotNAGConvergence(xPreF, allProbsParise, allProbsNoisyFP)
