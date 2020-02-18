import numpy as np

class QLearner:
    def __init__(self, r):
        self.gamma = .9
        self.tau   = 2
        self.alpha = 0.1 #self.tau * r


        self.qValues = np.zeros(2)
        self.qValues[0] = 1
        self.qValues[1] = np.log((0.3 * np.exp(self.tau  * self.qValues[0]))/0.7)/self.tau
        self.partitionFunction = np.sum([np.exp(self.tau * i) for i in self.qValues])
        self.actionProb = [np.exp(self.tau * i) / self.partitionFunction for i in self.qValues]
        self.action = 0
        self.payOffs = []

        self.gamesPlayed = 0

    def selectAction(self):
        self.partitionFunction = np.sum([np.exp(self.tau * i) for i in self.qValues])
        self.actionProb = [np.exp(self.tau * i) / self.partitionFunction for i in self.qValues]

        self.action = np.random.choice([0, 1], p=self.actionProb)

    def playGame(self, opponent, game):
        self.payOffs += [game[self.action, opponent.action, 0]]
        opponent.payOffs += [game[self.action, opponent.action, 1]]

        self.gamesPlayed += 1
        opponent.gamesPlayed += 1

    def qUpdate(self):

        self.payoff = np.mean(self.payOffs)
        self.qValues[self.action] += self.alpha * (self.payoff + self.gamma * max(self.qValues) - self.qValues[self.action])

        self.gamesPlayed = 0
