import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

dirName = 'ParameterSweep Results'

allmeanConvergence = np.zeros((10, 10))

i = 0
for r in tqdm(np.linspace(0, 5, num=10)):
    j = 0
    for gamma in np.linspace(-1, 1, num=10):
        with open(dirName + '/parameterSweep_r_{0}_gamma_{1}.txt'.format(r, gamma), 'r') as f:
            f.readline()
            line = f.readline()
            allmeanConvergence[i, 9 - j] = 1 - float(line[11:14])

        j+=1
    i+=1

plt.imshow(allmeanConvergence.T, cmap='hot', interpolation='nearest')
plt.show()