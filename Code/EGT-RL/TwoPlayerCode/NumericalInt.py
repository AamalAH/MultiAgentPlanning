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

nPlayers = 5
nActions = 50

dz = 1e-1
Zs = np.arange(-20, 20, dz)
Dz = (dz/np.sqrt(2 *  np.pi)) * np.exp(-(Zs**2)/2)


def getOrderParams(Xs, dz, Zs):
    q = np.sum((Xs ** 2) * dz)
    dx = np.array([Xs[i + 1] - Xs[i] for i in range(len(Zs[:-1]))])
    X = (1 / (q ** (nPlayers - 1)/2)) * np.sum((dx/dz) *dz)

    return q, X

def normaliseX(Xs):
    #Xs /= np.sum(abs(Xs))
    #return abs(Xs)
    return abs(abs(Xs)/np.sum(abs(Xs) * Dz))
    #return abs(abs(Xs)/np.sum(abs(Xs)))


allOrders = np.zeros((10, 10, 2))

idx1 = 0
for alpha in np.linspace(0.5, 1, num=1):
    idx2 = 1
    for Gamma in np.linspace(-1, 1, num=1):

        # alpha = 0.1
        tau = 1
        # Gamma = 1
        talpha = alpha/nActions
        ttau = tau/np.sqrt(nActions ** (nPlayers - 1))


        p = lambda xi, dz, Xs: np.sum((Xs * np.log(Xs/Xs[xi])) * dz)
        F = lambda xi, z, Xs, q, n, chi: (alpha**2) * (ttau ** 2) * Gamma * Xs[xi] * (q**(n-1)) * chi + alpha * ttau * (q**(n/2)) * z + talpha * tau * (q**((n+1)/2)) * z
        A = lambda xi, z, Xs, dz, q, n: -(q ** (n/2)) * z * (alpha * ttau + talpha * tau)

        Xs = normaliseX(np.random.random((len(Zs), )))
        Xd = normaliseX(np.random.random((len(Zs), )))
        
        #q, chi = getOrderParams(Xs, dz=dz, Zs=Zs)
        #qd, chid = getOrderParams(Xd, dz=dz, Zs=Zs)
        
        #checks = np.array([A(i, Zs[i], Xs, dz, q, nPlayers - 1) for i in range(len(Zs))])
        #checkd = np.array([A(i, Zs[i], Xd, dz, q, nPlayers - 1) for i in range(len(Zs))])
        
        #Xs *= (checks < 0)
        #Xd *= (checkd < 0)

        #Xs = normaliseX(Xs)
        #Xd = normaliseX(Xd)

        allF = []
        stopCond = False
        tol = 1e-5

        for i in range(1000):

            q, chi = getOrderParams(Xs, dz=dz, Zs=Zs)
            qd, chid = getOrderParams(Xd, dz=dz, Zs=Zs)

            fs = np.array([F(i, Zs[i], Xs, q, nPlayers - 1, chi) for i in range(len(Zs))])
            fd = np.array([F(i, Zs[i], Xd, qd, nPlayers - 1, chid) for i in range(len(Zs))])
            df = np.array([(fd[i] - fs[i])/(Xd[i] - Xs[i]) for i in range(len(Zs))])

            stopCond = np.max(fs) < tol
            allF.append(np.mean(fs))

            if ( stopCond ):
                break
            
            Xd = np.copy(Xs)
            #Xd = normaliseX(np.array([Xd[i] - (fd[i]/df[i]) for i in range(len(Zs))]))
            Xs = normaliseX(Xs - fs/df)
            checks = np.array([A(i, Zs[i], Xs, dz, q, nPlayers - 1) for i in range(len(Zs))])
            #Xs *= (checks < 0)
            Xs = normaliseX(Xs)


        allOrders[10 - idx2, idx1, 0] = chi
        allOrders[10 - idx2, idx1, 1] = q
        idx2 += 1
    idx1 += 1



