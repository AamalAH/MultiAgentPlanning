import numpy as np
from pydmd import DMD
import matplotlib.pyplot as plt

dim = 10

A = np.diag([np.random.rand() * np.random.choice([-1, 1]) for i in range(dim)])

x0 = np.random.rand(dim)

allSnapshots = [x0]

x = x0

for i in range(999):
    x = A @ x
    allSnapshots.append(x)
    

allSnapshots = np.array(allSnapshots).T


dmd = DMD(svd_rank=-1)

dmd.fit(allSnapshots)

plt.plot(dmd.reconstructed_data[0])
plt.plot(allSnapshots[0])
plt.show()
