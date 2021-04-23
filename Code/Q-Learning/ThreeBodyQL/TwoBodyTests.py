import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

alpha = 1

dxdt = lambda x:  alpha * x * (1 - x) * np.log(1 - x)
X = np.arange(0.01, 0.99, 0.001)
xdot = np.array([dxdt(x) for x in X])
plt.plot(X, xdot)
plt.show()