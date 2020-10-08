import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

alpha = .01
tau = 1

x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
z = np.linspace(0, 1, 50)

X, Y, Z = np.meshgrid(x, y, z)

A, B = np.array([[1, 5], [0, 3]]), np.array([[1, 0], [5, 3]])

E = lambda R: (R * np.log(R/R)) + ((1 - R) * np.log((1 - R)/R))

x_dot = lambda M, N, O: alpha * M * tau * ((A @ [N, 1-N])[0] - np.dot([M, 1-M], A@[N, 1-N])) + M * alpha * E(M)

y_dot = lambda M, N, O: alpha * N * tau * ((B.T @ [M, 1-M])[0] - np.dot([N, 1-N], B.T@[M, 1-M])) + N * alpha * E(N)

z_dot = lambda M, N, O: alpha * O * tau

U, V = np.zeros(X.shape), np.zeros(X.shape)
NI, NJ = X.shape

for i in range(NI):
    for j in range(NJ):
        U[i, j] = x_dot(X[i, j], Y[i, j])
        V[i, j] = y_dot(X[i, j], Y[i, j])

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111, projection='3d')

ax.quiver(X, Y, U, V, color = 'b', width=2e-3)
plt.xlabel('Probability of Action 1 (Agent 1)')
plt.ylabel('Probability of Action 1 (Agent 2)')
plt.show()
