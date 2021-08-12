import numpy as np
from scipy.linalg import block_diag
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

dS = np.genfromtxt('p_2_N_35/gamma_-0.11111111111111116_alpha_0.01')
print(dS.shape)
plt.plot(dS), plt.show()