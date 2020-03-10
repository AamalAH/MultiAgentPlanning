# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
from tqdm import tqdm

sys.path.append('../_NIPS2019_code/MAS_Environments')
sys.path.append('../_NIPS2019_code/MAS_Agents')
sys.path.append('../_NIPS2019_code/MAS12')

from Agent02 import Agent02
from Environment02 import Environment02
from GameAssignment01 import GameAssignment01
from MAgame1202 import runSim

dirName = 'ParameterSweep Results'
os.makedirs(dirName)

nSimulations = 1
gameName = 'PD'

allMeanTraj = []
allParams = []

converged = []
tau = 0.01

for r in tqdm(np.linspace(0, 1, num=100)):
    for gamma in np.linspace(-1, 0, num=100):

         parameters = {'tau': tau, 'lr': r * tau}
         
         allParams.append([tau, r])
         
         allxBar, converged = runSim(nSimulations, parameters, gamma)
         allMeanTraj.append(np.array(allxBar))

         # allMeanTraj.append(np.array(np.mean(np.array(allxBar).squeeze(), axis = 0)))

         with open(dirName + '/parameterSweep_r_{0}_gamma_{1}.txt'.format(r, gamma), 'w') as f:
            f.write('r: {0}, gamma: {1} \n'.format(r, gamma))
            f.write('Converged: {0} \n'.format(str(converged)))
            for probs in allMeanTraj[-1]:
                f.write(str(probs) + '\n')
            f.close()