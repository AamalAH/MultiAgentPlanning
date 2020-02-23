import numpy as np
from QLearner import QLearner
from tqdm import tqdm

nIter = 100
numberOfAgents = 500
numberofActions = 2
numberofOpponents = int(0.5 * numberOfAgents)

game = np.zeros((2, 2, 2))

game[:, :, 0] = np.array([[3, 0], [0, 1]])
game[:, :, 1] = np.array([[3, 5], [5, 1]])

agentPopulation = [QLearner(r=1) for i in range(numberOfAgents)]

for cIter in tqdm(range(nIter)):

    for i in range(numberOfAgents):
        agentPopulation[i].selectAction()

    for i in range(numberOfAgents):
        while agentPopulation[i].gamesPlayed < numberofOpponents:
            for m in range(numberofOpponents):
                o = i
                while o == i:
                    o = np.random.choice(list(range(10)))

                agentPopulation[i].playGame(agentPopulation[o], game)

    [agent.qUpdate() for agent in agentPopulation]

if __name__ == "__main__":
    rationality = np.linspace(0, 10, 0.1)
