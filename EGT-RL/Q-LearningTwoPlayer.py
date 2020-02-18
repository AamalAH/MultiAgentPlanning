from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

alpha = .1
gamma = .9

actions = ['defect', 'cooperate']

payoffA = np.array([[2, 3], [4, 1]])
payoffB = np.array([[3, 1], [2, 4]])

tau = 10

xsAll = []
for player in tqdm(range(int(4))):

    qValues = np.random.rand(2, 2)
    xs = []

    for cIter in tqdm(range(int(1e5))):

        partitionFunction = [np.sum([np.exp(tau * i) for i in qValues[p, :]]) for p in range(2)]
        actionProbs = np.array([[np.exp(tau * i)/partitionFunction[p] for i in qValues[p, :]] for p in range(2)])

        xs += [actionProbs[:, 0]]

        boltzmannChoices = [np.random.choice([0, 1], p=actionProbs[p, :]) for p in range(2)]

        rewards = [payoffA[boltzmannChoices[0], boltzmannChoices[1]], payoffB[boltzmannChoices[0], boltzmannChoices[1]]]

        qValues[0, boltzmannChoices[0]] += alpha * (rewards[0] - qValues[0, boltzmannChoices[0]] + gamma * max(qValues[0, :]))
        qValues[1, boltzmannChoices[1]] += alpha * (rewards[1] - qValues[1, boltzmannChoices[1]] + gamma * max(qValues[1, :]))

    xs = np.array(xs)

    xsAll += [xs]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

for X in xsAll:
    ax.plot(X[:, 0], X[:, 1])
    ax.scatter(X[0, 0], X[0, 1], color= 'r', marker='.')
    ax.scatter(X[-1, 0], X[-1, 1], color= 'r', marker='+')

plt.show()