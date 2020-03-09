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

nSimulations = 2
gameName = 'PD'

allMeanTraj = []
allParams = []

tau = 0.01
for r in tqdm(np.linspace(0, 1, num=100)):
        
         parameters = {'tau': tau, 'lr': r * tau}
         
         allParams.append([tau, r])
         
         allxBar = runSim(nSimulations, parameters, gameName)
         allMeanTraj.append(np.array(np.mean(np.array(allxBar).squeeze(), axis = 0)))

         with open(dirName + '/parameterSweep_tau_{0}_r_{1}.txt'.format(tau, r), 'w') as f:
            for probs in allMeanTraj[-1]:
                f.write(str(probs) + '\n')
            f.close()