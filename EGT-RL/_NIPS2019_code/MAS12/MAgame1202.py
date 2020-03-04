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
import datetime
import time
from tqdm import tqdm
sys.path.append('../MAS_Environments')
sys.path.append('../MAS_Agents')

from Environment02 import Environment02
#from Agent04 import Agent04
from Agent02 import Agent02

from GameAssignment01 import GameAssignment01

simStart = 1

Nsim = 100

T = 50
Nagent = 1000
Nplayer = 2
m = 50

Nact = 2

lr = 0.1
#epsilon = 0.2
tau = 2

gameName = 'PD'
#gameName = 'CE2'
#gameName = 'SH'
#gameName = 'HD'

initPara = '80,20,90,10'
#initPara = '20,80,80,20'
#initPara = '50,50,5,5'


dirName = 'result_Q-boltzmann_%s-%s_lr%.2f_temp%.1f_N%d_m%d'%(gameName, initPara, lr, tau, Nagent, m)

if not os.path.exists(dirName):
    os.makedirs(dirName)

reward1s = {}
reward2s = {}

# PD
reward1s['PD'] = np.array([[3, 0], [5, 1]])
reward2s['PD'] = np.array([[3, 5], [0, 1]])

# CE2
reward1s['CE2'] = -1*np.ones((Nact,Nact))+2*np.eye(Nact)
reward2s['CE2'] = -1*np.ones((Nact,Nact))+2*np.eye(Nact)

# SH
reward1s['SH'] = np.array([[1, 2], [0, 3]])
reward2s['SH'] = np.array([[1, 0], [2, 3]])

# HD
reward1s['HD'] = np.array([[1, 0], [2, -1]])
reward2s['HD'] = np.array([[1, 2], [0, -1]])

rMin = np.amin(reward1s[gameName])
rMax = np.amax(reward1s[gameName])
strInitPara = initPara.split(',')
intInitPara = [int(e) for e in strInitPara]

env = Environment02(Nact, reward1s[gameName], reward2s[gameName])
ga = GameAssignment01()

t1 = time.time()
for sim in tqdm(range(simStart, Nsim+1)):
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
        agent.Q[0] = x0*(rMax-rMin)+rMin
        agent.Q[1] = x1*(rMax-rMin)+rMin
        
        agents.append(agent)
    
    for t in range(T+1):
        agentsVS = ga.genAgentsVS(Nagent, m)
        
        #draw actions
        actions = []
        xbar = np.zeros(Nact, dtype=np.float)
        Qbar = np.zeros(Nact, dtype=np.float)
        
        for agent in agents:
            action = agent.getAction(0)
            actions.append(action)
            Qbar = Qbar + agent.Q
            xbar = xbar + agent.getProbS(0)
#            print(xbar)
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
    
    print('sim:', sim, 'countAction:', countAction)
    
    f01 = open(dirName+'/convergenceT-sim'+str(sim).zfill(3)+'.txt', 'w', encoding='utf-8')
    for convergence in convergenceT:
        f01.write(str(convergence)+'\n')
    f01.close()
    
    f01 = open(dirName+'/countActionT-sim'+str(sim).zfill(3)+'.txt', 'w', encoding='utf-8')
    for countAction in countActionT:
        s = [str(countAction[a]) for a in range(Nact)]
        f01.write(','.join(s)+'\n')
    f01.close()
    
    f01 = open(dirName+'/xbarT-sim'+str(sim).zfill(3)+'.txt', 'w', encoding='utf-8')
    for xbar in xbarT:
        f01.write(str(xbar)+'\n')
    f01.close()
    
    f01 = open(dirName+'/QbarT-sim'+str(sim).zfill(3)+'.txt', 'w', encoding='utf-8')
    for Qbar in QbarT:
        f01.write(str(Qbar)+'\n')
    f01.close()

t2 = time.time()
print('time:', t2-t1)
print('done',datetime.datetime.now())




















