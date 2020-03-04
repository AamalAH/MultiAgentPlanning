# -*- coding: utf-8 -*-

import sys
import numpy as np
from tqdm import tqdm

sys.path.append('../_NIPS2019_code/MAS_Environments')
sys.path.append('../_NIPS2019_code/MAS_Agents')
sys.path.append('../_NIPS2019_code/MAS12')

from Agent02 import Agent02
from Environment02 import Environment02
from GameAssignment01 import GameAssignment01
from MAgame1202 import runSim

nSimulations = 1
gameName = 'PD'

allMeanTraj = []

for tau in tqdm(np.linspace(0, 1, num = 2)):
    for lr in np.linspace(0, 1, num = 2):
        
         parameters = {'tau': tau, 'lr': lr}
         
         allxBar = runSim(nSimulations, parameters, gameName)
         allMeanTraj.append(np.array(allxBar))
         
 allMeanTraj = np.mean(np.array(allMeanTraj).squeeze(), axis = 0)