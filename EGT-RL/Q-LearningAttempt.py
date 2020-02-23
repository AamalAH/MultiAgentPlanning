from tqdm import tqdm

import numpy as np

alpha = .1
gamma = .9

trueValues = [24.4, 22.0, 19.8, 22.0, 19.8, 17.8, 19.8, 17.8, 16.0]

actions = ['up', 'down', 'left', 'right']

rewards = np.zeros((9, 9, 4))
rewards[0, 8, :] = 10
rewards[1, 1, 0] = rewards[2, 2, 0] = rewards[2, 2, 3] = rewards[3, 3, 2] = -1
rewards[5, 5, 3] = rewards[6, 6, 1] = rewards[6, 6, 2] = rewards[8, 8, 1] = -1
rewards[8, 8, 3] = rewards[1, 1, 0] = rewards[2, 2, 0] = rewards[2, 2, 3] = -1
rewards[3, 3, 2] = rewards[5, 5, 3] = rewards[6, 6, 1] = rewards[6, 6, 2] = -1
rewards[8, 8, 1] = rewards[8, 8, 3] = -1

transition = np.zeros((9, 9, 4))

transition[0, 8, 0] = transition[0, 8, 1] = transition[0, 8, 2] = transition[0, 8, 3] = 1
transition[1, 1, 0] = transition[1, 5, 1] = transition[1, 0, 2] = transition[1, 2, 3] = 1
transition[2, 2, 0] = transition[2, 5, 1] = transition[2, 1, 2] = transition[2, 2, 3] = 1
transition[3, 0, 0] = transition[3, 6, 1] = transition[3, 3, 2] = transition[3, 4, 3] = 1
transition[4, 1, 0] = transition[4, 7, 1] = transition[4, 3, 2] = transition[4, 5, 3] = 1
transition[5, 2, 0] = transition[5, 8, 1] = transition[5, 4, 2] = transition[5, 5, 3] = 1
transition[6, 3, 0] = transition[6, 6, 1] = transition[6, 6, 2] = transition[6, 7, 3] = 1
transition[7, 4, 0] = transition[7, 7, 1] = transition[7, 6, 2] = transition[7, 8, 3] = 1
transition[8, 5, 0] = transition[8, 8, 1] = transition[8, 7, 2] = transition[8, 8, 3] = 1

qValues = np.zeros((len(trueValues), 4))
s = 0

#Boltzmann action selection

tau = 0.1

for cIter in tqdm(range(int(1e6))):

    states, action = np.where(transition[s, :, :] != 0)

    partitionFunction = np.sum([np.exp(tau * i) for i in qValues[s, :]])
    actionProb = [np.exp(tau * i)/partitionFunction for i in qValues[s, :]]

    boltzmannChoice = np.random.choice(action, p=actionProb)
    a = action[boltzmannChoice]
    sPrime = states[boltzmannChoice]

    reward = rewards[s, sPrime, a]

    qValues[s, a] = qValues[s, a] + alpha * (reward + gamma * max(qValues[sPrime, :]) - qValues[s, a])

    s = sPrime


"""

# e-greedy action selection
e = 0.1
for cIter in range(int(1e6)):

    states, action = np.where(transition[s, :, :] != 0)
    if np.random.rand() <= e:
        a = np.random.randint(4)
        sPrime = states[a]
    else:
        greedyChoice = np.argmax(np.sum((transition[s, :, :] * qValues)[states], axis = 0))
        a = action[greedyChoice]
        sPrime = states[greedyChoice]

    reward = rewards[s, sPrime, a]

    qValues[s, a] = qValues[s, a] + alpha * (reward + gamma * max(qValues[sPrime, :]) - qValues[s, a])

    s = sPrime

"""
