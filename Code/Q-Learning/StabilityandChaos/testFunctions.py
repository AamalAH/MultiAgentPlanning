import numpy as np
from qUpdater.pPlayers import generateGames
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sps
from scipy.integrate import odeint
import itertools
from tqdm import tqdm


def TuylsODE(X0, t, payoffs, agentParams):

	alpha, tau, gamma = agentParams

	x = X0[0:2]
	y = X0[2:4]
	# z = X0[4:]

	xdot = np.zeros(2)
	ydot = np.zeros(2)
	# zdot = np.zeros(2)

	rewards = np.array([np.sum(np.reshape(payoffs[p], (nAct, nAct**(nPlayers - 1))) * np.array([(np.prod(X[T[np.where(T[:, p] == i + nAct*p)]], axis = 1)/X.reshape((nPlayers, nAct))[p, i]) for i in range(nAct)]), axis = 1) for p in range(nPlayers)])

	xdot[0] = x[0] * tau * (rewards[0, 0] - np.dot(x, rewards[0])) + alpha * x[0] * (x[1] * np.log(x[1]/x[0]))
	xdot[1] = x[1] * tau * (rewards[0, 1] - np.dot(x, rewards[0])) + alpha * x[1] * (x[0] * np.log(x[0]/x[1]))

	ydot[0] = y[0] * tau * (rewards[1, 0] - np.dot(y, rewards[1])) + alpha * y[0] * (y[1] * np.log(y[1]/y[0]))
	ydot[1] = y[1] * tau * (rewards[1, 1] - np.dot(y, rewards[1])) + alpha * y[1] * (y[0] * np.log(y[0]/y[1]))

	# zdot[0] = z[0] * tau * (rewards[2, 0] - np.dot(x, rewards[2])) + alpha * z[0] * (z[1] * np.log(z[1]/z[0]))
	# zdot[1] = z[1] * tau * (rewards[2, 1] - np.dot(x, rewards[2])) + alpha * z[1] * (z[0] * np.log(z[0]/z[1]))

	# return np.hstack((xdot, ydot, zdot))
	return np.hstack((xdot, ydot))
nPlayers = 2
nAct = 2

alpha = 0.1
tau  = 2

nIter = 1000

