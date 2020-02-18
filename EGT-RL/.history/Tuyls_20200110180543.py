import numpy as np
import matplotlib.pyplot as plt

alpha = 1
A = np.array([1, 5])
tau = 1

# pure strategies look like 1) (1, 0) and (1, 0) or 2) (1, 0) and (0, 1) or 3) (0, 1) and (1, 0) or 
# 4) (0, 1) and (0, 1)

# Then, the population will look like a mixture of all of these pure strategies. If we take the
# expectation on each of these, then that might end up looking like the mixed strategies that we're
# going for

# We will also need to define a population x to begin with so I'm not entirely sure where they got
# that from

x = np.linspace(0, 1, 50)

x_dot = lambda a: ((A[0] * a) - np.dot([a, 1-a], A@[a, 1-a])) * a

X_DOT = [x_dot(a) for a in x]

plt.plot(x, X_DOT, 'r-')
plt.show()