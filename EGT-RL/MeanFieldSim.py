import numpy as np
import matplotlib.pyplot as plt
from QLearner import QLearner
from tqdm import tqdm

nIter = 50
numberOfAgents = 1000
numberofActions = 2
numberofOpponents = int(0.05 * numberOfAgents)

game = np.zeros((2, 2, 2))

game[:, :, 0] = np.array([[1, 2], [0, 4]])
game[:, :, 1] = np.array([[1, 0], [2, 4]])

def runSim():

    meanQs = []
    actionProbs = []

    agentPopulation = [QLearner(r=1) for i in range(numberOfAgents)]

    for cIter in range(nIter):

        for i in range(numberOfAgents):
            agentPopulation[i].selectAction()

        for i in range(numberOfAgents):
            while agentPopulation[i].gamesPlayed < numberofOpponents:
                for m in range(numberofOpponents):
                    o = i
                    while o == i:
                        o = np.random.choice(list(range(10)))

                    agentPopulation[i].playGame(agentPopulation[o], game)

        meanQs += [np.mean(np.array([agentPopulation[i].qValues for i in range(numberOfAgents)]), axis=0)]
        actionProbs += [np.mean(np.array([np.array(agentPopulation[i].actionProb) for i in range(numberOfAgents)]), axis = 0)]

        [agent.qUpdate() for agent in agentPopulation]

    return actionProbs

if __name__ == "__main__":
    nSimulations = 50

    actionProbEvolutions = []

    for citer in tqdm(range(nSimulations)):

        actionProbEvolutions += [np.array(runSim())]

    meanEvolutions  = sum(actionProbEvolutions)/nSimulations