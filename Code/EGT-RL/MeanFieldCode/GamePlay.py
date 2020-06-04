import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from MeanFieldCode.QLearner import QLearner
from MeanFieldCode.Game import Game

# Generate Random Games
# Initialise players
# Iterate Game
    # Choose actions
    # Receive rewards
    # Update Q-values

def generateGame(gamma, nAct):
    reward1s = {}
    reward2s = {}

    nElements = nAct**2 #number of payoff elements in the matrix

    cov = np.eye(2 *  nElements)
    cov[:nElements, nElements:] = np.eye(nElements) * gamma
    cov[nElements:, :nElements] = np.eye(nElements) * gamma

    rewards = np.random.multivariate_normal(np.zeros(2 * nElements), cov=cov)

    rewards = np.array([2, 0, 0, 1, 1, 0, 0, 2])

    reward1s = rewards[0:nElements].reshape((nAct, nAct))
    reward2s = rewards[nElements:].reshape((nAct, nAct)).T

    return Game([reward1s, reward2s])

def initialisePlayers(params, nAgent, nAct, game):
    return [QLearner(params, nAct, game.rewards[i], nAgent-1) for i in range(nAgent)]

def selectActions(players):
    [player.selectAction() for player in players]
    return players

def playGame(players, game):
    nAgents = len(players)
    # players = np.array(players)
    game.generateGames(players, players[0].nOpponents)
    for p in players:
        p.payOffs = []

    for p in range(nAgents):
        players = game.playGame(players, p)

    return players.tolist()

def updateQValues(players):
    [player.qUpdate() for player in players]
    return players


def iterateGame(players, game):
    players = selectActions(players)
    players = playGame(players, game)
    players = updateQValues(players)

    return players

def runSimulation(params, nAgent, nAct):
    game = generateGame(params[2], nAct)
    players = initialisePlayers(params, nAgent, nAct, game)

    nIter = int(1e3)

    probs = []

    for cIter in tqdm(range(nIter)):
        players = iterateGame(players, game)
        probs.append(np.array([p.actionProb[0] for p in players]))

    return players, probs

def runTests(paramsAgent, paramsEnvironment):

    nTests = 10

    allResults = []

    for cTest in range(nTests):
        players, probs = runSimulation(paramsAgent, paramsEnvironment[0], paramsEnvironment[1])
        allResults.append(np.array(probs))

    return players, allResults

def plotResults(allResults):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    for result in allResults:
        # ax1.scatter(results[0, 0], probs[0, 1], marker='o')
        ax1.plot(result[:, 0], result[:, 1], 'k--', zorder=1)

    for result in allResults:
        ax1.scatter(result[-1, 0], result[-1, 1], color='r', marker='+', zorder=2)

    plt.show()

if __name__ == "__main__":
    """
    AgentParameters = [alpha, tau, gamma]
    Environment Parameters = [nAgent, nAction]
    """

    alpha = 0.1
    tau = 2
    gamma = 0

    nAgent = 2
    nAct = 2

    paramsAgent = np.array([tau, alpha, gamma])
    paramsEnvironment = [nAgent, nAct]

    _, allResults = runTests(paramsAgent, paramsEnvironment)

    plotResults(allResults)

