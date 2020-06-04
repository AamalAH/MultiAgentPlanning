import random

class Game:

    def __init__(self, rewards):
        self.rewards = rewards
        self.allGames = np.eye((len(rewards)))

    def generateOpponents(self, p):
        # players = np.array(range(nAgents))
        # np.random.shuffle(players[np.where(gamesPlayed < nOpponents)])
        return np.where(self.allGames[:, p] == 1)[0].tolist()
        # return players[0:nOpponents - gamesPlayed[p]]

    def generateGames(self, players, nOpponents):
        nAgents = len(players)
        self.allGames = np.zeros((nAgents, nAgents))
        for p in range(nAgents):
            availableOpponents = np.array(range(nAgents))[np.where((np.array(range(nAgents)) != p) & (self.allGames[:, p] != 1) & (np.sum(self.allGames, axis = 1) < nOpponents))].tolist()
            opponents = random.sample(availableOpponents, nOpponents - len(np.where(self.allGames[:,  p] == 1)[0]))
            self.allGames[opponents, p] = 1
            self.allGames[p, opponents] = 1

    def playGame(self, players, P):
        players = np.array(players)
        opponents = self.generateOpponents(P)
        players[P].payOffs += self.rewards[0][players[P].action, [players[O].action for O in opponents]].tolist()
        for O in opponents:
            players[O].payOffs += [self.rewards[1][players[P].action, players[O].action]]
            self.allGames[P, O], self.allGames[O, P] = 0, 0
        return players

if __name__ == "__main__":
    from MeanFieldCode.QLearner import QLearner
    from MeanFieldCode.GamePlay import *

    alpha = 0.1
    tau = 1
    gamma = 0

    nAgent = 2
    nAct = 2

    game = generateGame(gamma, nAct)

    players = [QLearner([alpha, tau, gamma], nAct, game.rewards[i], nAgent - 1) for i in range(nAgent)]
    selectActions(players)

    game.generateGames(players, nAgent-1)
    game.playGame(players, 0)
    print('hi')
