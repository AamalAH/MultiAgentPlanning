# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:24:15 2020

@author: aamal
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

nPlayers = 3
nActions = 10
nIter = 10
dz = 1e-3
Zs = np.arange(-1.5, 1.5, dz)
Dz = (dz/np.sqrt(2 *  np.pi)) * np.exp(-(Zs**2)/2)
step = 1e-3


allOrders = []

for a in np.linspace(0.222, 1, num=1):
    for G in np.linspace(-1, 0, num=10):

        t = 10
        
        ta = a/nActions
        tt = t/np.sqrt(nActions ** (nPlayers - 1))
        th = t/np.sqrt(nActions**nPlayers)
        
        p = lambda xi, Xs: np.sum((Xs * np.log(Xs/Xs[xi])) * Dz)
        F = lambda xi, z, Xs, q, n, chi: a**2 * tt**2 * G * chi * q**(n-1) * Xs[xi] + q**(n/2) * z * (a * tt + a * th)**2 + ta * p(xi, Xs)
        # F = lambda xi, z, Xs, q, n, chi: (a**2) * (tt**2) * G * chi * (q**(n-1)) * Xs[xi] + a * tt * q**(n/2) * z + ta * t * q**((n + 1)/2) * z + ta * p(xi, Xs)
        
        
        def getOrderParams(Xs, dz, Zs):
            q = np.sum((Xs ** 2) * Dz)
            dx = np.array([Xs[i + 1] - Xs[i] for i in range(len(Zs[:-1]))])
            X = (1 / (q ** (nPlayers - 1)/2)) * np.sum((dx/dz) * Dz[:-1])
        
            return q, X
        
        def normaliseX(Xs):
            return abs(Xs)/np.sum(abs(Xs) * Dz)
        
        def getF(Xs):
            qs, chis = getOrderParams(Xs, dz, Zs)
            return np.array([F(i, Zs[i], Xs, qs, nPlayers - 1, chis) for i in range(len(Xs))])
        
        
        Xs = normaliseX(np.ones((len(Zs), )))
        XsOld = np.copy(Xs)
        Xsmin = np.copy(Xs)
        fs = np.ones((len(Xs), ))
        fsMin = np.copy(fs)
        allF = []
        for cIter in tqdm(range(500)):
        
            if (cIter % 10 == 0) and cIter != 0:
                allF += [np.max(abs(fs))] 
                
                if np.max(abs(fs)) < np.max(abs(fsMin)):
                    Xsmin = np.copy(Xs)
                    fsMin = np.copy(fs)
                
                else:
                    Xs = np.copy(Xsmin)
                    fs = np.copy(fsMin)
                    continue
        
            diff = 1e-7 * np.random.normal(0, 1, len(Zs))
            Xd = Xs + diff
            Xdm = Xs - diff
        
            fsOld = np.copy(fs)
            fs = getF(Xs)
            
        
            fd = getF(Xd)
            fdm = getF(Xdm)
        
            df = (fd - fdm)/(Xd - Xdm)
        
            XsOld = np.copy(Xs)
            Xs -= step * (fs/df)
            Xs = normaliseX(nActions * (Xs/max(Xs)))
    
        if np.max(abs(fs)) < np.max(abs(fsMin)):
            Xsmin = np.copy(Xs)
            fsMin = np.copy(fs)
        
        else:
            Xs = np.copy(Xsmin)
            fs = np.copy(fsMin)
            
        qs, chis = getOrderParams(Xs, dz, Zs)
        allOrders.append([a, G, qs, chis])
