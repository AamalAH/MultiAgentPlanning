# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:24:15 2020

@author: aamal
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

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
for a in tqdm(np.linspace(1e-2, 3e-2, num=1)):
    for G in np.linspace(-1, 0, num=1):
        a = a
        b = 5e-2 * np.sqrt(nActions)
        T = a / b

        F = lambda xi, z, Xs, q, n, chi: (p-1)*G * chi * Xs[xi] - T * np.log(Xs[xi]) + np.sqrt((p-1) * q) * z

        def getOrderParams(Xs, dz, Zs):
            q = np.sum(((Xs) ** 2) * Dz)
            dx = np.array([(Xs[i + 1] - Xs[i]) for i in range(len(Zs[:-1]))])
            X = (1 / np.sqrt((p-1) * q)) * np.sum(Xs * Zs * Dz)

            return q, X

        def normaliseX(Xs):
            return Xs / np.sum(Xs * Dz)

        def getF(Xs):
            qs, chis = getOrderParams(Xs, dz, Zs)
            return np.array([F(i, Zs[i], Xs, qs, nPlayers - 1, chis) for i in range(len(Xs))])

        Xs = normaliseX(np.random.rand(len(Zs)))
        XsOld = np.copy(Xs)
        Xsmin = np.copy(Xs)
        fs = np.ones((len(Xs),))
        fsOld = np.copy(fs)
        # allF = np.zeros((len(Xs), 1000))
        for cIter in (range(5000)):

            if np.any(Xs < 0):
                Xs = np.copy(XsOld)
            # if abs(fs[150]) > abs(fsOld[150]):
            #     Xs = np.copy(XsOld)

            diff = 1e-7
            Xd = Xs + diff
            Xdm = Xs - diff

            fsOld = np.copy(fs)
            fs = getF(Xs)
            # allF[:, cIter] = fs
            fd = getF(Xd)
            fdm = getF(Xdm)

            df = 0.5 * (((fd - fs) / (Xd - Xs)) + ((fs - fdm) / (Xs - Xdm)))

            XsOld = np.copy(Xs)
            Xs -= step * (fs / df)
            Xs = normaliseX(Xs)

        # plt.plot(allF[150, :]), plt.show()
        qs, chis = getOrderParams(Xs, dz, Zs)
        allOrders.append([a, G, qs, chis])
        allA[cL] = np.sum(Dz * (1/np.abs((a / (b * Xs)) - (p - 1) * G * chis)**2))
        cL += 1

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
print(getF(Xs))