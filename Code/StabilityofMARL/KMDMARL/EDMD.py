import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
from itertools import product
"""
EDMD Algorithm: 

1. Define a series of observables: let's say Hermite Polynomials up to order 5 + [x, v] + [1]
2. Collect a set of data pairs X_n-1 and X_n
3. Apply the observable functions on the datasets to get \Phi(X), \Phi(Y)
4. Compute G and A from EDMD paper
5. Compute K

Predicting the flow field:

Apply K on the observable for some initial condition. 
Extract the flow field using (3.2) and (3.3) of the KMD for Algs paper
"""

def eval_Phi(x, maxOrder = 5):
    lenData = len(x)
    allHermiteOrders = list(product(range(maxOrder), repeat=lenData))
    allHermites = np.array([sps.eval_hermite(o,x) for o in range(maxOrder)])

    allCombos = [np.prod(allHermites[oList, range(lenData)]) for oList in allHermiteOrders]

    return allCombos

def generatePhi(X, maxOrder=5):
    return [eval_Phi(x, maxOrder=maxOrder) for x in X]