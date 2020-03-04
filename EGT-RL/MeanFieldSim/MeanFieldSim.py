import numpy as np
import random
import matplotlib.pyplot as plt
from QLearner import QLearner
from tqdm import tqdm

nIter = 50
numberOfAgents = 1000
numberofActions = 2
numberofOpponents = int(0.05 * numberOfAgents)

game = np.zeros((2, 2, 2))

game[:, :, 0] = np.array([[3, 5], [0, 1]])
game[:, :, 1] = np.array([[3, 0], [5, 1]])

def runSim():

    meanQs = []
    actionProbs = []

    agentPopulation = np.array([QLearner(r=1, nOpponents=numberofOpponents) for i in range(numberOfAgents)])

    for cIter in range(nIter):

        [agentPopulation[i].selectAction() for i in range(numberOfAgents)]
        gamesPlayed = np.zeros(numberOfAgents).astype(int)

        for i in range(numberOfAgents):
            while gamesPlayed[i] < numberofOpponents:

                availableOpponents = np.array(range(numberOfAgents))
                availableOpponents = availableOpponents[np.where(availableOpponents != i)]

                opponentIdx = random.sample(list(availableOpponents), numberofOpponents - gamesPlayed[i])

                agentPopulation[i].playGame(agentPopulation[opponentIdx], game)

                gamesPlayed[i] += len(opponentIdx)
                gamesPlayed[opponentIdx] += 1

        meanQs += [np.mean(np.array([agentPopulation[i].qValues for i in range(numberOfAgents)]), axis=0)]
        actionProbs += [np.mean(np.array([np.array(agentPopulation[i].actionProb) for i in range(numberOfAgents)]), axis = 0)]

        [agent.qUpdate() for agent in agentPopulation]

    return actionProbs

if __name__ == "__main__":
    nSimulations = 5

    actionProbEvolutions = []

    for citer in tqdm(range(nSimulations)):
        actionProbEvolutions += [np.array(runSim())]

    meanEvolutions = sum(actionProbEvolutions) / nSimulations