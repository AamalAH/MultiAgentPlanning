# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:40:55 2019

@author: cwleung
"""
# concurrent learning (2p), 1 vs m
# no network
# matrix game
# Q

import sys
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from tqdm import tqdm
sys.path.append('../MAS_Environments')
sys.path.append('../MAS_Agents')

from Environment02 import Environment02
from Agent02 import Agent02

from GameAssignment01 import GameAssignment01


def runSim(Nsim, parameters, gamma):
    
    T = 50
    Nagent = 2
    m = 1
    
    Nact = 2
    
    tau = parameters['tau']
    lr = parameters['lr']

    initPara = '50,50,5,5'
    
    reward1s = {}
    reward2s = {}

    cov = np.eye(8)
    cov[:4, 4:] = np.eye(4) * gamma
    cov[4:, :4] = np.eye(4) * gamma

    rewards = np.random.multivariate_normal(np.zeros(8), cov=cov)

    reward1s = rewards[0:4].reshape((2, 2))
    reward2s = rewards[4:].reshape((2, 2)).T
    
    rMin = np.amin(reward1s)  #min and max rewards from the game, will help in defining the initial Q distribution
    rMax = np.amax(reward1s)
    strInitPara = initPara.split(',')
    intInitPara = [int(e) for e in strInitPara]
    
    env = Environment02(Nact, reward1s, reward2s)
    ga = GameAssignment01()
    
    allxBar = []
    
    # This will run multiple simulations so that the results can be averaged over
    allConverged = []
    for sim in range(Nsim):
        countActionT = []
        convergenceT = []
        xbarT = []
        QbarT = []
        
        #initialize
        agents = []
        for i in range(Nagent):
            agent = Agent02(env,
                            lr=lr,
                            tau=tau)
            
            x0 = np.random.beta(intInitPara[0],intInitPara[1])
            x1 = np.random.beta(intInitPara[2],intInitPara[3])
            agent.Q[0] = x0*(rMax-rMin)+rMin # scales the distribution to ensure that the Q0 values lie within the given range
            agent.Q[1] = x1*(rMax-rMin)+rMin
            
            agents.append(agent)

        maxT = 1e5
        xbar = np.zeros(Nact, dtype=np.float)

        stopCond = False
        tol = 1e-6
        t = 0

        allStep = []

        while t < maxT and not stopCond:

            agentsVS = ga.genAgentsVS(Nagent, m) # assigns the opponents to the agents
            
            #draw actions
            actions = []

            oldxBar = np.copy(xbar)

            xbar = np.zeros(Nact, dtype=np.float)
            Qbar = np.zeros(Nact, dtype=np.float)
            
            for agent in agents:
                action = agent.getAction(0)
                actions.append(action)
                Qbar = Qbar + agent.Q
                xbar = xbar + agent.getProbS(0)

            actions = np.array(actions)
            Qbar = Qbar / Nagent
            xbar = xbar / Nagent
            
            #recording
            countAction = Counter(actions)
            convergence = max(countAction.values())/Nagent
            countActionT.append(countAction)
            convergenceT.append(convergence)
            xbarT.append(xbar)
            QbarT.append(Qbar)
            
            #play games
            for i, agent in enumerate(agents):
                subActions = actions[agentsVS[i]]
                countSubAction = Counter(subActions)
                
                avgReward = 0
                for a in range(Nact):
                    moves = [actions[i], a]
                    avgReward += countSubAction[a]*env.getRewards(moves)[0]
                avgReward /= m
                
                agent.train(0, actions[i], avgReward, 0)
        
            normStep = np.linalg.norm((xbar - oldxBar)/oldxBar)

            allStep.append(normStep)

            stopCond = normStep < tol
            t += 1

        print('r: {0}, gamma: {1}, Simulation: {2}, Converged: {3}'.format(lr/tau, gamma, sim, stopCond))
        allConverged.append(stopCond)
        allxBar.append(np.array(xbarT))

    return allxBar #np.mean(allConverged)


if __name__ == "__main__":
    nSim = 2
    gamma = -.3
    
    parameters = {'tau': 0.01, 'lr': 0.005}
    
    allxBar = runSim(nSim, parameters, gamma)
    
    # meanTraj = np.mean(np.array(allxBar).squeeze(), axis = 0)
    # plt.plot(meanTraj)
    plt.show()
    
    














