# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:24:15 2020

@author: aamal
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from Agent import generateGames


F = lambda xi, z, Xs, q, n, chi, rho, b, a: (b**2)*G * chi * Xs[xi] - a * np.log(Xs[xi]) + b * np.sqrt(q) * z - rho

def getOrderParams(Xs, dz, Zs):
    q = np.sum(((Xs) ** 2) * Dz)
    dx = np.array([(Xs[i + 1] - Xs[i]) for i in range(len(Xs[:-1]))])
    dxdz = dx / np.array([(Zs[i + 1] - Zs[i]) for i in range(len(Zs[:-1]))])
    X = (1 / np.sqrt(q)) * np.sum(dxdz * Dz[:-1])

    return q, X

def normaliseX(Xs):
    return Xs / np.sum(Xs * Dz)

def getF(Xs, rho, b, a):
    qs, chis = getOrderParams(Xs, dz, Zs)
    return np.array([F(i, Zs[i], Xs, qs, nPlayers - 1, chis, rho, b, a) for i in range(len(Xs))])

def lineSearch(xs, b, a, step = 1e-5, R=np.linspace(-0.5, 0.5, num=100)):
    allYs = np.zeros(np.shape(R))
    diff = 1e-7

    for i in range(len(R)):
        xd = xs + diff
        xdm = xs - diff

        fs = getF(xs, rho + R[i], b, a)
        fd = getF(xd, rho + R[i], b, a)
        fdm = getF(xdm, rho + R[i], b, a)

        df = 0.5 * (((fd - fs) / (xd - xs)) + ((fs - fdm) / (xs - xdm)))

        allYs[i] = abs(1 - np.sum((xs - step * fs/df) * Dz))

    return rho + R[np.argmin(allYs)]

nPlayers = 2
p = nPlayers
nActions = 5
# nIter = 10
dz = 1e-2
Zs = np.arange(-1.5, 1.5, dz)
Dz = (dz / np.sqrt(2 * np.pi)) * np.exp(-(Zs ** 2) / 2)
step = 1e-5

allOrders = []
allA = np.zeros(400)

cL = 0
for a in np.linspace(1e-2, 3e-2, num=1):
    for G in np.linspace(-1, 0, num=1):
        a = a
        b = 5e-2 * np.sqrt(nActions)

        Xs = normaliseX(np.random.rand(len(Zs)))
        A, _ = generateGames(G, 1, nActions)
        A = np.squeeze(A)
        rho = (1/nActions) * np.dot(np.ones(nActions) * (1/nActions), A @ np.ones(nActions) * (1/nActions)) - a * np.log(1/nActions)

        Xsmin = np.copy(Xs)
        fs = np.ones((len(Xs),))

        allF = np.zeros((len(Xs), 500))
        for cIter in tqdm(range(500)):
            if cIter == 498:
                print('hi')
            rho = lineSearch(Xs, b, a)

            diff = 1e-7
            Xd = Xs + diff
            Xdm = Xs - diff

            fsOld = np.copy(fs)
            fs = getF(Xs, rho, b, a)
            fd = getF(Xd, rho, b, a)
            fdm = getF(Xdm, rho, b, a)

            df = 0.5 * (((fd - fs) / (Xd - Xs)) + ((fs - fdm) / (Xs - Xdm)))

            XsOld = np.copy(Xs)
            
            Xs -= step * (fs / df)
            Xs = normaliseX(Xs)

            if abs(getF(Xs, rho, b, a)[0]) > abs(fs[0]):
                Xs = np.copy(XsOld)

            allF[:, cIter] = getF(Xs, rho, b, a)

        # # plt.plot(allF[150, :]), plt.show()
        # qs, chis = getOrderParams(Xs, dz, Zs)
        # allOrders.append([a, G, qs, chis])
        # allA[cL] = np.sum(Dz * (1/np.abs((a / (b * Xs)) - (p - 1) * G * chis)**2))
        # cL += 1

# tt = lambda t: t / np.sqrt(nActions ** (nPlayers - 1))
# th = lambda t: t / np.sqrt(nActions ** nPlayers)
# ta = lambda a: a / nActions
#
#
# vals = np.array([abs((a ** 2) * (tt(t) ** 2) * G * q ** (nPlayers - 2) * chi) ** (2) - (
#             2 * (a * tt(t) + a * th(t)) ** 2 * (nPlayers - 1) * q ** (nPlayers - 2)) for (a, G, q, chi) in allOrders])
# b = np.flip(vals.reshape((20, 20)).T, axis=0)
# sns.heatmap(b, xticklabels=np.linspace(1, 5, num=20), yticklabels=p.linspace(-1, 0, num=20)[-1::-1]), plt.xlabel(
#     'alpha x 1e2'), plt.ylabel('Gamma'), plt.show()
# print(getF(Xs, rho))