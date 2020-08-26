import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# q = int Dz x^2
# 1 = int Dz x
# X = 1/q int dx/dz

# initialise values of z
# for every z initialise a guess of x and x_d where x_d differs from x by some delta
# use these choices for x to determine X, p, q
# evaluate f and f' at this guess (f' given by finite approximation with x_d)
# update x using Newton Raphson with all the finite approximations

# determine q and X off that

def getOrderParams(Xs, dz, Zs):
    q = np.sum((Xs ** 2) * dz)
    dx = np.array([Xs[i + 1] - Xs[i] for i in range(len(Zs[:-1]))])
    X = (1 / q) * np.sum((dx/dz) * dz)

    return q, X

def normaliseX(Xs):
    return abs(abs(Xs)/np.sum(abs(Xs)))

nPlayers = 3
nActions = 2

allOrders = np.zeros((10, 10, 2))

idx1 = 0
for alpha in np.linspace(0, 1, num=10):
    idx2 = 1
    for Gamma in np.linspace(-1, 1, num=10):

        # alpha = 0.1
        tau = 1
        # Gamma = 1
        talpha = alpha/nActions
        ttau = tau/np.sqrt(nActions ** (nPlayers - 1))

        dz = 1e-1

        Zs = np.arange(-20, 20, dz)

        p = lambda xi, dz, Xs: np.sum((Xs * np.log(Xs/Xs[xi])) * dz)
        F = lambda xi, z, Xs, q, n, chi: (alpha**2) * (ttau ** 2) * Gamma * Xs[xi] * (q**(n-1)) * chi + alpha * ttau * (q**(n/2)) * z + talpha * tau * (q**((n+1)/2)) * z + talpha * p(xi, dz, Xs)

        Xs = normaliseX(np.random.random((len(Zs), )))
        Xd = normaliseX(np.random.random((len(Zs), )))

        allF = []
        stopCond = False
        tol = 1e-5


        for i in range(1000):

            q, chi = getOrderParams(Xs, dz=dz, Zs=Zs)
            qd, chid = getOrderParams(Xd, dz=dz, Zs=Zs)
            fs = np.array([F(i, Zs[i], Xs, q, nPlayers - 1, chi) for i in range(len(Zs))])

            allF.append(np.mean(fs))

            stopCond = np.max(fs) < tol

            fd = np.array([F(i, Zs[i], Xd, qd, nPlayers - 1, chid) for i in range(len(Zs))])
            df = np.array([(fd[i] - fs[i])/(Xd[i] - Xs[i]) for i in range(len(Zs))])

            if ( stopCond ):
                break

            Xd = np.copy(Xs)
            Xs = normaliseX(np.array([Xs[i] - (fs[i]/df[i]) for i in range(len(Zs))]))

        allOrders[10 - idx2, idx1, 0] = chi
        allOrders[10 - idx2, idx1, 1] = q
        idx2 += 1
    idx1 += 1

