import numpy as np
from pClass import p
from tqdm import tqdm

## Set up code parameters
tau = 10

## Set up game
game = np.zeros((2, 2, 2))
game[:, :, 0] = np.array([[3, 0], [5, 1]])
game[:, :, 1] = np.array([[3, 5], [0, 1]])

nAgents = 1e3
qValues = np.random.rand(2, int(nAgents))
meanStrategy = [np.mean([np.exp(tau * qValues[0, i])/(np.exp(tau * qValues[0, i]) + np.exp(tau * qValues[1, i])) for i in range(int(nAgents))]), np.mean([np.exp(tau * qValues[1, i])/(np.exp(tau * qValues[0, i]) + np.exp(tau * qValues[1, i])) for i in range(int(nAgents))])]

## Generate pInit

P = p(game, meanStrategy=meanStrategy, initP=True, initQ=qValues)

for t in tqdm(np.arange(0, 0.5, P.deltaT)):

    meanStrategy = np.zeros(2)
    for q1 in np.arange(0, 1, P.deltaQ):
        for q2 in np.arange(0, 1, P.deltaQ):
            meanStrategy[0] += np.exp(P.tau * (q1 + P.deltaQ/2))/(np.exp(P.tau * (q1 + P.deltaQ/2)) + np.exp(P.tau * (q2 + P.deltaQ/2))) * P([q1, q2])

            meanStrategy[1] += np.exp(P.tau * (q2 + P.deltaQ/2))/(np.exp(P.tau * (q1 + P.deltaQ/2)) + np.exp(P.tau * (q2 + P.deltaQ/2))) * P([q1, q2])
    print(sum(meanStrategy))

    P = p(game, oldP=P, meanStrategy=meanStrategy)