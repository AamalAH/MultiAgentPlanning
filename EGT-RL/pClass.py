import numpy as np


class p():
    def __init__(self, game, oldP = None,  meanStrategy=None, initP = False, initQ = None):
        self.tau = 10
        self.eta = 1
        self.game = game

        self.deltaT = 1e-2
        self.deltaQ = 1e-2

        self.oldP = oldP
        self.initP = initP
        self.initQ = initQ
        self.x = meanStrategy
        if self.initP:
            self.v = lambda j, Q, x: 0
        else:
            self.v = lambda j, Q, x: self.eta * np.exp(self.tau * Q[j])/(np.exp(self.tau * Q[j]) + np.exp(self.tau * Q[1 - j])) * ((self.game[:, :, 0] @ x)[j] - Q[j])

    def __call__(self, Q):
        if (self.initP):
            return len(np.where((self.initQ[0, :] < Q[0] + self.deltaQ) & (self.initQ[0, :] > Q[0]) & (self.initQ[1, :] < Q[1] + self.deltaQ) & (self.initQ[1, :] > Q[1]))[0])/self.initQ.shape[1]
            
        else:
            return self.oldP(Q) + self.oldP.pDot(Q) * self.deltaT

    def pDot(self, Q):
        if self.initP:
            return 0
        else:
            return -1 * (self.oldP([Q[0] + self.deltaQ, Q[1]]) * self.oldP.v(0, [Q[0] + self.deltaQ, Q[1]], self.oldP.x) -
                     self.oldP(Q) * self.oldP.v(0, Q, self.oldP.x))/self.deltaQ - \
                    (self.oldP([Q[0], Q[1] + self.deltaQ]) * self.oldP.v(1, [Q[0], Q[1] + self.deltaQ], self.oldP.x) -
                     self.oldP(Q) * self.oldP.v(1, Q, self.oldP.x))/self.deltaQ

