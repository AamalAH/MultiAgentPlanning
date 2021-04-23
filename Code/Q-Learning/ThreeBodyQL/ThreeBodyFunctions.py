import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sps
from scipy.integrate import odeint
from tqdm import tqdm

def generateRandomGames():
    
    C = np.random.multivariate_normal(np.zeros(4), np.eye(4)).reshape((2, 2))
    Z = np.random.multivariate_normal(np.zeros(4), np.eye(4)).reshape((2, 2))
    
    return C, Z

def generateMatchingPennies():
    """
    Create Matching Pennies Matrix

    :return:
    """

    C = np.array([[5, -5], [-5, 5]])
    Z = np.array([[1, -1], [-1, 1]])

    return C, Z

def getActionProbs(Q, agentParams):
    """
        qValues: nPlayer x nActions x nSim
        return: nPlayer x nActions x nSim
        """
    alpha, tau, gamma = agentParams

    return np.exp(tau * Q) / np.sum(np.exp(tau * Q), axis=1)[:, None]

def getCurrentActions(actionProbs):
    return [np.random.choice([0, 1], p=actionProbs[p, :]) for p in range(3)]

def getRewards(G, bChoice):
    A, B = G

    rewards = np.zeros(3)
    rewards[0] = A[bChoice[0], bChoice[1]] + B[bChoice[2], bChoice[0]]
    rewards[1] = B[bChoice[0], bChoice[1]] + A[bChoice[1], bChoice[2]]
    rewards[2] = A[bChoice[2], bChoice[0]] + B[bChoice[1], bChoice[2]]

    return rewards

def qUpdate(Q, G, agentParams):

    alpha, tau, gamma = agentParams

    actionProbs = getActionProbs(Q, agentParams)
    bChoice = getCurrentActions(actionProbs)
    rewards = getRewards(G, bChoice)

    for p in range(3):
        Q[p, bChoice[p]] += alpha * (rewards[p] - Q[p, bChoice[p]] + gamma * np.max(Q[p, :]))
    return Q

def initialiseQ():
    return np.random.rand(3, 2)

def simulate(agentParams, nIter = 5e3):
    nIter = int(nIter)

    G = generateMatchingPennies()
    Q = initialiseQ()

    firstActionTracker = np.zeros((3, nIter))

    for cIter in range(nIter):
        Q = qUpdate(Q, G, agentParams)
        firstActionTracker[:, cIter] = getActionProbs(Q, agentParams)[:, 0]

    return firstActionTracker

def TuylsZZC(X, t, G, agentParams):

    C, Z = G

    alpha, tau, gamma = agentParams

    x = X[0:2]
    y = X[2:4]
    z = X[4:]

    xdot = np.zeros(2)
    ydot = np.zeros(2)
    zdot = np.zeros(2)

    xdot[0] = alpha * x[0] * tau * ((Z @ y)[0] + (Z @ z)[0] - np.dot(x, (Z @ y) + (Z @ z))) + alpha * x[0] * (x[1] * np.log(x[1]/x[0]))
    xdot[1] = alpha * x[1] * tau * ((Z @ y)[1] + (Z @ z)[1] - np.dot(x, (Z @ y) + (Z @ z))) + alpha * x[1] * (x[0] * np.log(x[0]/x[1]))

    ydot[0] = alpha * y[0] * tau * ((C @ z)[0] + ((-Z).T @ x)[0] - np.dot(y, (C @ z) + ((-Z).T @ x))) + alpha * y[0] * (y[1] * np.log(y[1]/y[0]))
    ydot[1] = alpha * y[1] * tau * ((C @ z)[1] + ((-Z).T @ x)[1] - np.dot(y, (C @ z) + ((-Z).T @ x))) + alpha * y[1] * (y[0] * np.log(y[0]/y[1]))

    zdot[0] = alpha * z[0] * tau * (((-Z).T @ x)[0] + (C.T @ y)[0] - np.dot(z, ((-Z).T @ x) + (C.T @ y))) + alpha * z[0] * (z[1] * np.log(z[1]/z[0]))
    zdot[1] = alpha * z[1] * tau * (((-Z).T @ x)[1] + (C.T @ y)[1] - np.dot(z, ((-Z).T @ x) + (C.T @ y))) + alpha * z[1] * (z[0] * np.log(z[0]/z[1]))

    return np.hstack((xdot, ydot, zdot))

def TuylsMismatching(X, t, M, agentParams):

    C = np.array([[0, 1], [M, 0]])

    # C = generateRandomGames()[0]

    alpha, tau, gamma = agentParams

    x = X[0:2]
    y = X[2:4]
    z = X[4:]

    xdot = np.zeros(2)
    ydot = np.zeros(2)
    zdot = np.zeros(2)

    xdot[0] = alpha * x[0] * tau * ((C @ z)[0] - np.dot(x, (C @ z))) + alpha * x[0] * (x[1] * np.log(x[1]/x[0]))
    xdot[1] = alpha * x[1] * tau * ((C @ z)[1] - np.dot(x, (C @ z))) + alpha * x[1] * (x[0] * np.log(x[0]/x[1]))

    ydot[0] = alpha * y[0] * tau * ((C @ x)[0] - np.dot(y, (C @ x))) + alpha * y[0] * (y[1] * np.log(y[1]/y[0]))
    ydot[1] = alpha * y[1] * tau * ((C @ x)[1] - np.dot(y, (C @ x))) + alpha * y[1] * (y[0] * np.log(y[0]/y[1]))

    zdot[0] = alpha * z[0] * tau * ((C @ y)[0] - np.dot(z, (C @ y))) + alpha * z[0] * (z[1] * np.log(z[1]/z[0]))
    zdot[1] = alpha * z[1] * tau * ((C @ y)[1] - np.dot(z, (C @ y))) + alpha * z[1] * (z[0] * np.log(z[0]/z[1]))

    return np.hstack((xdot, ydot, zdot))

def TuylsChakraborty(X, t, chakrabortyCase, agentParams):

    if chakrabortyCase == 1:
        S, T = -2, 3
    elif chakrabortyCase == 2:
        S, T = 0.5, 1.5
    elif chakrabortyCase == 3:
        S, T = 1.5, 2.5
    elif chakrabortyCase == 4:
        S, T = 2.5, 1.5
    elif chakrabortyCase == 5:
        S, T = -2, 0.5
    elif chakrabortyCase == 6:
        S, T = 0.4, 0.6
    elif chakrabortyCase == 7:
        S, T = 0.6, 0.4
    elif chakrabortyCase == 8:
        S, T = 1.5, 0.5
    elif chakrabortyCase == 9:
        S, T = -2, -0.5
    elif chakrabortyCase == 10:
        S, T = -0.5, -2
    elif chakrabortyCase == 11:
        S, T = 0.5, -2
    elif chakrabortyCase == 12:
        S, T = 3, -2

    C = np.array([[1, S], [T, 0]])

    alpha, tau, gamma = agentParams

    x = X[0:2]
    y = X[2:4]
    z = X[4:]

    xdot = np.zeros(2)
    ydot = np.zeros(2)
    zdot = np.zeros(2)

    xdot[0] = alpha * x[0] * tau * ((C @ z)[0] - np.dot(x, (C @ z))) + alpha * x[0] * (x[1] * np.log(x[1]/x[0]))
    xdot[1] = alpha * x[1] * tau * ((C @ z)[1] - np.dot(x, (C @ z))) + alpha * x[1] * (x[0] * np.log(x[0]/x[1]))

    ydot[0] = alpha * y[0] * tau * ((C @ x)[0] - np.dot(y, (C @ x))) + alpha * y[0] * (y[1] * np.log(y[1]/y[0]))
    ydot[1] = alpha * y[1] * tau * ((C @ x)[1] - np.dot(y, (C @ x))) + alpha * y[1] * (y[0] * np.log(y[0]/y[1]))

    zdot[0] = alpha * z[0] * tau * ((C @ y)[0] - np.dot(z, (C @ y))) + alpha * z[0] * (z[1] * np.log(z[1]/z[0]))
    zdot[1] = alpha * z[1] * tau * ((C @ y)[1] - np.dot(z, (C @ y))) + alpha * z[1] * (z[0] * np.log(z[0]/z[1]))

    return np.hstack((xdot, ydot, zdot))

def TuylsCCZ(X, t, G, agentParams):

    C, Z = G

    alpha, tau, gamma = agentParams

    x = X[0:2]
    y = X[2:4]
    z = X[4:]

    xdot = np.zeros(2)
    ydot = np.zeros(2)
    zdot = np.zeros(2)

    xdot[0] = alpha * x[0] * tau * ((C @ y)[0] + (C @ z)[0] - np.dot(x, (C @ y) + (C @ z))) + alpha * x[0] * (x[1] * np.log(x[1]/x[0]))
    xdot[1] = alpha * x[1] * tau * ((C @ y)[1] + (C @ z)[1] - np.dot(x, (C @ y) + (C @ z))) + alpha * x[1] * (x[0] * np.log(x[0]/x[1]))

    ydot[0] = alpha * y[0] * tau * ((Z @ z)[0] + (C.T @ x)[0] - np.dot(y, (Z @ z) + (C.T @ x))) + alpha * y[0] * (y[1] * np.log(y[1]/y[0]))
    ydot[1] = alpha * y[1] * tau * ((Z @ z)[1] + (C.T @ x)[1] - np.dot(y, (Z @ z) + (C.T @ x))) + alpha * y[1] * (y[0] * np.log(y[0]/y[1]))

    zdot[0] = alpha * z[0] * tau * ((C.T @ x)[0] + ((-Z).T @ y)[0] - np.dot(z, (C.T @ x) + ((-Z).T @ y))) + alpha * z[0] * (z[1] * np.log(z[1]/z[0]))
    zdot[1] = alpha * z[1] * tau * ((C.T @ x)[1] + ((-Z).T @ y)[1] - np.dot(z, (C.T @ x) + ((-Z).T @ y))) + alpha * z[1] * (z[0] * np.log(z[0]/z[1]))

    return np.hstack((xdot, ydot, zdot))

if __name__ == '__main__':


    chakrabortyCase = 2

    G = generateRandomGames()
    # G = generateMatchingPennies()
    M = 2

    fig = plt.figure()
    

    initConds = [np.array([0.41030259, 0.58969741, 0.56361327, 0.43638673, 0.53308779,
       0.46691221]), np.array([8.00418146e-01, 1.99581854e-01, 2.93321817e-04, 9.99706678e-01,
       1.12708184e-01, 8.87291816e-01]), np.array([0.50203898, 0.49796102, 0.88428137, 0.11571863, 0.03295509,
       0.96704491]), np.array([0.44888546, 0.55111454, 0.34212499, 0.65787501, 0.33387894,
       0.66612106]), np.array([0.92317559, 0.07682441, 0.89499898, 0.10500102, 0.04261427,
       0.95738573]), np.array([0.72500583, 0.27499417, 0.04536833, 0.95463167, 0.60778619,
       0.39221381]), np.array([0.80655268, 0.19344732, 0.91123289, 0.08876711, 0.89704441,
       0.10295559]), np.array([0.97353206, 0.02646794, 0.38587598, 0.61412402, 0.1708327 ,
       0.8291673 ]), np.array([0.46419295, 0.53580705, 0.74290291, 0.25709709, 0.7391003 ,
       0.2608997 ]), np.array([0.75603941, 0.24396059, 0.50732423, 0.49267577, 0.33829168,
       0.66170832])]

    # for i, tau in tqdm(enumerate(np.linspace(50, 100, num=5))):   
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax = fig.add_subplot(1, 5, i+1, projection='3d')
    ax.set_xlabel('P1')
    ax.set_ylabel('P2')
    ax.set_zlabel('P3')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])                       
                           
    for x0 in initConds:

      # x0 = np.random.rand(3)
      # x0 = (np.vstack((x0, 1 - x0)).T).reshape(6)
      # alpha, tau, gamma = np.random.rand(), np.random.randint(1, 11), 0.1
      alpha, tau, gamma = 0.1, 0.001, 0.1
      agentParams = alpha, tau, gamma

      t = np.linspace(0, int(1e6), int(1e8) + 1)

      sol = odeint(TuylsCCZ, x0, t, args=(G, agentParams))
      # sol = odeint(TuylsZZC, x0, t, args=(G, agentParams))

      # sol = odeint(TuylsMismatching, x0, t, args=(M, agentParams))
      # sol = odeint(TuylsChakraborty, x0, t, args=(chakrabortyCase, agentParams))
      ax.plot(sol[:, 0], sol[:, 2], sol[:, 4])
      ax.scatter(sol[0, 0], sol[0, 2], sol[0, 4], marker='o', color='r')
      ax.scatter(sol[-1, 0], sol[-1, 2], sol[-1, 4], marker='+', color='k')

          # ax.title.set_text('alpha = {0}, tau = {1}'.format(np.round(alpha, 2), np.round(tau, 2)))
  
    plt.show()