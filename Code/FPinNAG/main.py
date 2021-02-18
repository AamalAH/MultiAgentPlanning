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


def updateSigma(adjMat, vectors):
    return (adjMat @ vectors.T).T

def NAGProcedure(adjMat, initQ, initVectors, initRef, xPreF=None, process="Parise"):
    Q = initQ
    sigmaRef = initRef
    probVectors = initVectors

    cIter = 0
    stopCond = False

    # allProbs = []

    while not stopCond:
        allProbs += [probVectors]

        probVectors = np.array([bestResponse(i, sigmaRef, Q, xPreF)[0] for i in range(3)]).T

        # averageProbs = (1 / len(allProbs)) * np.sum(np.array(allProbs), axis=0)

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

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.show()


if __name__ == "__main__":
    (adjMat, initQ, initVectors, initRef) = initialise()

    xPreF = np.random.rand(2, 3)
    xPreF /= xPreF.sum(axis=0)

    allProbsParise = NAGProcedure(adjMat, initQ, initVectors, initRef, xPreF, process="Parise")
    allProbsFP = NAGProcedure(adjMat, initQ, initVectors, initRef, xPreF, process="FP")

    plotNAGConvergence(xPreF, allProbsParise)
